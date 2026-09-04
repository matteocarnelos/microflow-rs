//! Generates golden fixtures for the `conv_1d` kernel.
//!
//! The expected outputs come from a naive reference implementation of the
//! spec semantics (int8 dot product, int32 accumulator, per-channel requant
//! with `round_ties_even`, saturating cast, optional ReLU clamped at the
//! output zero point). The reference is intentionally written independently
//! from the kernel: `tests/conv1d_golden.rs` compares the kernel against
//! these fixtures bit-by-bit.
//!
//! Regenerate with:
//!
//! ```text
//! cargo run --example golden_gen
//! ```
//!
//! The RNG seed is fixed: same seed -> identical fixtures.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Fixed seed: fixtures are reproducible and committed to the repo.
const SEED: u64 = 42;

/// Number of random data variants per shape.
const VARIANTS_PER_SHAPE: usize = 8;

/// Shapes cover the spec ranges: T 1-64, C 1-8, K 1-7, stride 1-2,
/// valid/same, including T < K with same padding.
/// (T, C, K, F, stride, same)
const SHAPES: &[(usize, usize, usize, usize, usize, bool)] = &[
    (5, 1, 3, 1, 1, false),
    (5, 1, 3, 1, 1, true),
    (4, 1, 2, 2, 1, false),
    (8, 2, 3, 4, 1, true),
    (8, 2, 3, 4, 2, true),
    (16, 4, 3, 4, 2, false),
    (7, 8, 5, 8, 1, true),
    (64, 8, 7, 16, 2, true),
    (2, 1, 3, 1, 1, true),
    (1, 1, 1, 1, 1, false),
    (3, 2, 1, 2, 2, false),
    (63, 1, 7, 1, 2, false),
];

struct Case {
    t: usize,
    c: usize,
    k: usize,
    f: usize,
    stride: usize,
    same: bool,
    relu: bool,
    scale_x: f32,
    zp_x: i8,
    scale_out: f32,
    zp_out: i8,
    filter_scales: Vec<f32>,
    input: Vec<i8>,
    weights: Vec<i8>,
    bias: Vec<i32>,
}

impl Case {
    fn generate(shape: (usize, usize, usize, usize, usize, bool), rng: &mut StdRng) -> Self {
        let (t, c, k, f, stride, same) = shape;
        let zp = |rng: &mut StdRng| rng.random_range(-16..=16);
        let scale = |rng: &mut StdRng| 0.005 + rng.random::<f32>() * 0.045;
        Self {
            t,
            c,
            k,
            f,
            stride,
            same,
            relu: rng.random_bool(0.25),
            scale_x: scale(rng),
            zp_x: zp(rng),
            scale_out: scale(rng),
            zp_out: zp(rng),
            // Weights zero point is 0, as in the spike model (spec F5)
            filter_scales: (0..f)
                .map(|_| 0.001 + rng.random::<f32>() * 0.009)
                .collect(),
            input: (0..t * c).map(|_| rng.random_range(-128..=127)).collect(),
            weights: (0..f * k * c)
                .map(|_| rng.random_range(-128..=127))
                .collect(),
            bias: (0..f).map(|_| rng.random_range(-5000..=5000)).collect(),
        }
    }

    fn out_len(&self) -> usize {
        if self.same {
            self.t.div_ceil(self.stride)
        } else {
            (self.t - self.k) / self.stride + 1
        }
    }

    /// Naive reference: mirrors the spec formulas, not the kernel code.
    fn expected_output(&self) -> Vec<i8> {
        let out_len = self.out_len();
        let pad_total = ((out_len - 1) * self.stride + self.k).saturating_sub(self.t);
        let pad_left = pad_total / 2;
        let mut out = Vec::with_capacity(out_len * self.f);
        for t in 0..out_len {
            for f in 0..self.f {
                let mut acc: i32 = 0;
                for m in 0..self.k {
                    let index = t * self.stride + m;
                    let index = index as isize - pad_left as isize;
                    if !(0..self.t as isize).contains(&index) {
                        continue;
                    }
                    let index = index as usize;
                    for ch in 0..self.c {
                        let sample = i32::from(self.input[index * self.c + ch]);
                        let weight = i32::from(self.weights[(f * self.k + m) * self.c + ch]);
                        acc += (sample - i32::from(self.zp_x)) * weight;
                    }
                }
                let raw = acc + self.bias[f];
                let multiplier = (self.scale_x * self.filter_scales[f]) / self.scale_out;
                let requantized =
                    (raw as f32 * multiplier).round_ties_even() + f32::from(self.zp_out);
                let mut value = requantized as i8; // saturating cast
                if self.relu {
                    value = value.max(self.zp_out); // ReLU in quantized coordinates
                }
                out.push(value);
            }
        }
        out
    }

    fn render(&self, expected: &[i8]) -> String {
        let mut text = format!(
            "case {} {} {} {} {} {} {}\n",
            self.t,
            self.c,
            self.k,
            self.f,
            self.stride,
            if self.same { "same" } else { "valid" },
            if self.relu { "relu" } else { "none" }
        );
        text += &format!(
            "quant {:?} {} {:?} {}\n",
            self.scale_x, self.zp_x, self.scale_out, self.zp_out
        );
        text += &format!(
            "filter_scales {}\n",
            self.filter_scales
                .iter()
                .map(|s| format!("{s:?}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // Input: one row of C values per timestep
        text += "input\n";
        for t in 0..self.t {
            text += &self.row(&self.input[t * self.c..(t + 1) * self.c]);
        }
        // Weights: F blocks of K rows of C values
        text += "weights\n";
        for f in 0..self.f {
            for m in 0..self.k {
                let start = (f * self.k + m) * self.c;
                text += &self.row(&self.weights[start..start + self.c]);
            }
        }
        text += &format!(
            "bias {}\n",
            self.bias
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        // Expected: one row of F values per output timestep
        text += "expected\n";
        let out_len = self.out_len();
        for t in 0..out_len {
            text += &self.row(&expected[t * self.f..(t + 1) * self.f]);
        }
        text
    }

    fn row(&self, values: &[i8]) -> String {
        format!(
            "{}\n",
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut text = String::from(
        "# microflow conv_1d golden fixtures (seed 42)\n\
         # DO NOT EDIT: regenerate with `cargo run --example golden_gen`\n",
    );
    let mut count = 0;
    for shape in SHAPES {
        for _ in 0..VARIANTS_PER_SHAPE {
            let case = Case::generate(*shape, &mut rng);
            let expected = case.expected_output();
            text += &case.render(&expected);
            count += 1;
        }
    }
    let path = "tests/golden/conv1d.txt";
    std::fs::write(path, text).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("written {count} cases to {path}");
}
