// SPDX-License-Identifier: Apache-2.0
//! The megakernel's interpreter (FP8 S8 M0, `kernels/mk_controller.cu`) on the H100: its
//! persistent loop, instruction ring, pages and cross-SM counters, played from hand-written
//! instruction lists (M0 builds them in the test; M1 has `#[forward]` emit them).
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips. Every test arms
//! the watchdog (`common::watchdog`), so a counter wait that never ends fails its test instead of
//! hanging the binary.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod watchdog;
}

use scratchy_target_cuda::kernels::{
    KernelError, MK_INSTRUCTION_WIDTH, MK_OP_CHAIN, MK_OP_NOOP, MK_OP_PAGE_ECHO, MK_PAGE_FLOATS,
    MK_WORKERS, mk_test,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open().expect("an H100: these tests run only where one is (GKE)")
}

/// `[MK_WORKERS, n, MK_INSTRUCTION_WIDTH]` i32, every instruction `OP_NOOP`.
struct Lists {
    n: usize,
    words: Vec<i32>,
}

impl Lists {
    fn new(n: usize) -> Self {
        Self {
            n,
            words: vec![MK_OP_NOOP; MK_WORKERS * n * MK_INSTRUCTION_WIDTH],
        }
    }

    /// Worker `w`'s instruction `i`: `op` and its argument words.
    fn set(&mut self, w: usize, i: usize, op: i32, args: &[i32]) {
        let at = (w * self.n + i) * MK_INSTRUCTION_WIDTH;
        self.words[at] = op;
        self.words[at + 1..at + 1 + args.len()].copy_from_slice(args);
    }

    fn bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

fn i32s(bytes: &[u16]) -> Vec<i32> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|p| i32::from(p[0]) | (i32::from(p[1]) << 16))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bits: &[u16]) -> Vec<f32> {
    (bits.as_chunks::<2>().0.iter())
        .map(|p| f32::from_bits(u32::from(p[0]) | (u32::from(p[1]) << 16)))
        .collect()
}

/// One play of `lists` over fresh zeroed counters and chain, and `x` (`rows` rows); returns the
/// chain and `out`.
fn play(
    device: &CudaDevice,
    lists: &Lists,
    (counters, chain): (usize, usize),
    (x, rows): (&[f32], usize),
) -> Result<(Vec<i32>, Vec<f32>), KernelError> {
    let up = |b: &[u8]| DeviceBuffer::upload_bytes(device, b).expect("upload");
    let ins = up(&lists.bytes());
    let cb = up(&vec![0u8; 4 * counters]);
    let chb = up(&vec![0u8; 4 * chain]);
    let xb = up(&f32_bytes(x));
    let ob = up(&f32_bytes(&vec![f32::NAN; rows * MK_PAGE_FLOATS]));
    // SAFETY: every buffer holds its count of elements; every index the lists name is inside its
    // buffer, and every count waited for is published (the tests build them so); null is the
    // default stream, which the downloads wait for.
    unsafe {
        mk_test(
            (ins.ptr().cast_const(), lists.n as u32),
            (cb.ptr(), counters as u32),
            (chb.ptr(), chain as u32),
            (xb.ptr().cast_const(), ob.ptr(), rows as u32),
            std::ptr::null_mut(),
        )
    }?;
    Ok((
        i32s(&chb.download().expect("chain")),
        f32s(&ob.download().expect("out")),
    ))
}

/// M0 Day 1: a chain over every worker of every SM, in an order that crosses SMs at every link
/// (worker `w`'s link is worker `perm(w)`'s successor, `perm` a stride of 67 mod 264), several
/// links per worker, each waiting on the one before it through a counter. `chain[j] = j + 1` for
/// every link proves the ring delivered each instruction, the dispatch ran it, every acquire saw
/// the release before it, and nothing ran twice; a missed publish hangs, and the watchdog fails
/// the test. Then each lane's pages: every worker echoes a row of its own through each of its 6
/// pages, and `out` is `x`, bit for bit. Then the refusals.
#[test]
fn the_interpreter_plays_a_cross_sm_chain_and_its_pages() {
    let device = h100();
    let mut pass = true;
    // The chain: link `j` (0..LINKS) runs on worker `order[j % W]`, as its instruction `j / W`.
    const PER: usize = 3;
    let links = MK_WORKERS * PER;
    let order: Vec<usize> = (0..MK_WORKERS).map(|w| (w * 67) % MK_WORKERS).collect();
    let mut seen = vec![false; MK_WORKERS];
    order.iter().for_each(|&w| seen[w] = true);
    assert!(
        seen.iter().all(|&s| s),
        "67 is coprime with 264: every worker once"
    );
    let mut lists = Lists::new(PER);
    for j in 0..links {
        let (w, i) = (order[j % MK_WORKERS], j / MK_WORKERS);
        // Wait for counter `j − 1` (link j − 1 published it), read chain[j − 1], write chain[j],
        // publish counter `j`. Link 0 waits for nothing.
        let (wait_idx, want, read) = match j {
            0 => (0, 0, 0),
            _ => (j as i32 - 1, 1, j as i32 - 1),
        };
        lists.set(
            w,
            i,
            MK_OP_CHAIN,
            &[wait_idx, want, read, j as i32, j as i32],
        );
    }
    let rows = 1;
    let x = vec![0.0f32; rows * MK_PAGE_FLOATS];
    match play(&device, &lists, (links, links), (&x, rows)) {
        Ok((chain, _)) => {
            let bad = (chain.iter().enumerate())
                .filter(|&(j, &v)| v != j as i32 + 1)
                .count();
            eprintln!(
                "M0 interpreter chain: {links} links over {MK_WORKERS} workers, bad_links={bad} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            pass &= bad == 0;
        }
        Err(e) => {
            eprintln!("M0 interpreter chain: launch failed: {e} FAIL");
            pass = false;
        }
    }
    // The pages: worker `w` echoes row `6w + p` through its page `p`, for p in 0..6, one
    // instruction each, the lane's two workers per SM running at once.
    const PAGES: usize = 6;
    let rows = MK_WORKERS * PAGES;
    let x: Vec<f32> = (0..rows * MK_PAGE_FLOATS)
        .map(|i| (i as f32) * 0.5 - 7.25)
        .collect();
    let mut lists = Lists::new(PAGES);
    for w in 0..MK_WORKERS {
        for p in 0..PAGES {
            let row = (w * PAGES + p) as i32;
            lists.set(w, p, MK_OP_PAGE_ECHO, &[row, p as i32, row]);
        }
    }
    match play(&device, &lists, (1, 1), (&x, rows)) {
        Ok((_, out)) => {
            let bad = (out.iter().zip(&x))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "M0 interpreter pages: {rows} rows through {PAGES} pages per lane, \
                 mismatched_bits={bad} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            pass &= bad == 0;
        }
        Err(e) => {
            eprintln!("M0 interpreter pages: launch failed: {e} FAIL");
            pass = false;
        }
    }
    // The refusals, before any pointer is touched.
    let null: *mut std::ffi::c_void = std::ptr::null_mut();
    for dims in [[0u32, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]] {
        // SAFETY: the launcher refuses these counts before it touches a pointer.
        let got = unsafe {
            mk_test(
                (null.cast_const(), dims[0]),
                (null, dims[1]),
                (null, dims[2]),
                (null.cast_const(), null, dims[3]),
                null,
            )
        };
        let ok = got == Err(KernelError::MkShape { dims });
        eprintln!(
            "M0 interpreter refuses {dims:?}: {got:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(pass, "the interpreter failed: see the lines above");
}
