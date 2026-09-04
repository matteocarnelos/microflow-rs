use core::array;

use libm::truncf;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D, TensorViewPadding};

/// Round half to even (banker's rounding, spec §3.1).
///
/// `f32::round_ties_even` is std-only and the inherent float methods are
/// sparse in this no_std core build, so the semantics are re-created from
/// `libm::truncf` — bit-identical results, pinned by the golden fixtures.
fn round_ties_even(value: f32) -> f32 {
    let truncated = truncf(value);
    let fraction = value - truncated;
    let magnitude = fraction.abs();
    if magnitude < 0.5 {
        truncated
    } else if magnitude > 0.5 {
        truncated + fraction.signum()
    } else {
        // Exactly .5: pick the even neighbor (truncated is even -> stay).
        if truncf(truncated / 2.0) * 2.0 == truncated {
            truncated
        } else {
            truncated + fraction.signum()
        }
    }
}

/// Options of the Conv1D operator.
pub struct Conv1DOptions {
    pub fused_activation: FusedActivation,
    pub padding: TensorViewPadding,
    /// Stride along the time axis.
    pub stride: usize,
}

/// Number of output timesteps for the given geometry.
///
/// Returns `None` when the geometry is not representable: with `Valid`
/// padding the kernel must fit inside the input (`TIMESTEPS >= KERNEL`),
/// because empty outputs are forbidden.
const fn output_len(
    timesteps: usize,
    kernel: usize,
    stride: usize,
    padding: TensorViewPadding,
) -> Option<usize> {
    match padding {
        TensorViewPadding::Valid => {
            if timesteps >= kernel {
                Some((timesteps - kernel) / stride + 1)
            } else {
                None
            }
        }
        TensorViewPadding::Same => Some(timesteps.div_ceil(stride)),
    }
}

/// Left padding of the "same" scheme for the given output position.
///
/// Mirrors the TFLite convention: the total padding is split with the
/// extra sample on the right.
const fn same_pad_left(output_len: usize, stride: usize, kernel: usize, timesteps: usize) -> usize {
    let pad_total = ((output_len - 1) * stride + kernel).saturating_sub(timesteps);
    pad_total / 2
}

/// Performs the Conv1D operation on a rank-4 tensor with a single row,
/// i.e. input shape `(1, 1, TIMESTEPS, CHANS)` — the layout TFLite uses
/// to serialize Keras `Conv1D` layers.
///
/// Quantization is per-channel on the filters:
///
/// ```text
/// acc    = Σ (x - zp_x) · (w_f - zp_w_f)     over the kernel window
/// raw    = acc + bias_f                       bias in accumulator units
/// out    = saturate(round_ties_even(raw · m_f) + zp_out)
/// m_f    = (scale_x · scale_w_f) / scale_out
/// ```
///
/// With `Same` padding the out-of-range positions contribute nothing to
/// the accumulator: they behave as if filled with `zp_x`.
///
/// # Arguments
/// * `input` - The input tensor of shape `(1, 1, TIMESTEPS, CHANS)`
/// * `filters` - The filters of OHWI shape `(FILTERS, 1, KERNEL, CHANS)`,
///   with per-channel scales and zero points (`FILTERS_QUANTS`)
/// * `biases` - The biases in accumulator units, shape `(FILTERS, 1)`
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`Conv1DOptions`] struct
///
pub fn conv_1d<
    T: Quantized,
    const TIMESTEPS: usize,
    const CHANS: usize,
    const KERNEL: usize,
    const FILTERS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_TIMESTEPS: usize,
>(
    input: Tensor4D<T, 1, 1, TIMESTEPS, CHANS, 1>,
    filters: &Tensor4D<T, FILTERS, 1, KERNEL, CHANS, FILTERS_QUANTS>,
    biases: &Tensor2D<i32, FILTERS, 1, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv1DOptions,
) -> Tensor4D<T, 1, 1, OUTPUT_TIMESTEPS, FILTERS, 1> {
    debug_assert_eq!(
        output_len(TIMESTEPS, KERNEL, options.stride, options.padding),
        Some(OUTPUT_TIMESTEPS),
        "OUTPUT_TIMESTEPS must match the padding geometry"
    );
    let pad_left = match options.padding {
        TensorViewPadding::Same => {
            same_pad_left(OUTPUT_TIMESTEPS, options.stride, KERNEL, TIMESTEPS)
        }
        TensorViewPadding::Valid => 0,
    };
    let input_zero_point = i32::from_subset(&input.zero_point[0]);
    let output_zero_point_f32 = f32::from_subset(&output_zero_point[0]);
    // Per-channel requant multipliers, computed once per filter
    let multipliers: [f32; FILTERS] = array::from_fn(|f| {
        let filter_scale = filters.scale.get(f).copied().unwrap_or(filters.scale[0]);
        (input.scale[0] * filter_scale) / output_scale[0]
    });
    // Zero-point hoisting (week 6): the spec §3.1 accumulator
    //     sum_m (x - zx)(w - zw)
    // expands exactly (integer algebra, no rounding moves) to
    //     sum x*w - zx * sum w - zw * sum x + M * zx * zw,
    // with M the in-window element count. The MAC loop drops to a bare
    // multiply-accumulate; the corrections are O(F + T) instead of
    // O(T * F * K * C) — the golden fixtures pin the outputs bit-for-bit.
    //
    // Prefix sums over the kernel axis (inclusive): `prefix[f][j]` is the
    // weight sum of kernel positions `0..=j`. The out-of-bounds positions
    // of a Same-padding edge window form a contiguous prefix/suffix, so
    // the in-window weight sum is a difference of two prefix lookups.
    let filter_weight_prefix: [[i32; KERNEL]; FILTERS] = array::from_fn(|f| {
        let mut prefix = [0i32; KERNEL];
        let mut running = 0i32;
        for (m, slot) in prefix.iter_mut().enumerate() {
            running += (0..CHANS)
                .map(|c| i32::from_subset(&filters.buffer[f][(0, m)][c]))
                .sum::<i32>();
            *slot = running;
        }
        prefix
    });
    let output = [Buffer2D::from_fn(|_, t| {
        // In-bounds kernel window [m_lo, m_hi): out-of-bounds positions
        // (Same padding at the edges) contribute nothing, spec §3.2 — for
        // Valid padding this is always the full kernel.
        let first = t * options.stride;
        let m_lo = (pad_left.saturating_sub(first)).min(KERNEL);
        let m_hi = (TIMESTEPS + pad_left - first).clamp(0, KERNEL);
        // The full-window branch keeps the loop bounds compile-time constants
        // (0..KERNEL) so LLVM unrolls the kernel axis — the dynamic range is
        // only for Same-padding edge windows. Both branches compute the same
        // sums; only the bounds differ.
        let full_window = m_lo == 0 && m_hi == KERNEL;
        let chan_sum = |index: usize| -> i32 {
            (0..CHANS)
                .map(|c| i32::from_subset(&input.buffer[0][(0, index)][c]))
                .sum()
        };
        // Per-window input sum: shared by every filter (the zw correction).
        let window_sum: i32 = if full_window {
            (0..KERNEL).map(|m| chan_sum(first + m - pad_left)).sum()
        } else {
            (m_lo..m_hi).map(|m| chan_sum(first + m - pad_left)).sum()
        };
        array::from_fn(|f| {
            let filter_zero_point = i32::from_subset(
                &filters
                    .zero_point
                    .get(f)
                    .copied()
                    .unwrap_or(filters.zero_point[0]),
            );
            // Dot product of the in-bounds kernel window with filter `f`
            // (the full-window branch keeps static bounds for unrolling).
            let mut dot: i32 = 0;
            if full_window {
                for m in 0..KERNEL {
                    let index = first + m - pad_left;
                    for c in 0..CHANS {
                        let sample = i32::from_subset(&input.buffer[0][(0, index)][c]);
                        let weight = i32::from_subset(&filters.buffer[f][(0, m)][c]);
                        dot += sample * weight;
                    }
                }
            } else {
                for m in m_lo..m_hi {
                    let index = first + m - pad_left;
                    for c in 0..CHANS {
                        let sample = i32::from_subset(&input.buffer[0][(0, index)][c]);
                        let weight = i32::from_subset(&filters.buffer[f][(0, m)][c]);
                        dot += sample * weight;
                    }
                }
            }
            // The hoisted zero-point corrections (exact; see above). The
            // in-window weight sum via the prefix lookups (0 -> empty sum).
            let below = |j: usize| {
                if j == 0 {
                    0
                } else {
                    filter_weight_prefix[f][j - 1]
                }
            };
            let in_window_weights = below(m_hi) - below(m_lo);
            let in_window_count = ((m_hi - m_lo) * CHANS) as i32;
            let accumulator =
                dot - input_zero_point * in_window_weights - filter_zero_point * window_sum
                    + in_window_count * input_zero_point * filter_zero_point;
            // Requantize: round-to-nearest-even, then saturate on cast
            let raw = f32::from_subset(&(accumulator + biases.buffer[(f, 0)]));
            let requantized = round_ties_even(raw * multipliers[f]) + output_zero_point_f32;
            let value = T::from_superset_unchecked(&requantized);
            match options.fused_activation {
                FusedActivation::None => value,
                FusedActivation::Relu => relu(value, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(value, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `(1, 1, T, C)` input from a flat row-major slice.
    fn input<const T: usize, const C: usize>(
        values: &[i8],
        scale: f32,
        zero_point: i8,
    ) -> Tensor4D<i8, 1, 1, T, C, 1> {
        Tensor4D::new(
            [Buffer2D::from_fn(|_, t| {
                core::array::from_fn(|c| values[t * C + c])
            })],
            [scale],
            [zero_point],
        )
    }

    /// Builds `(F, 1, K, C)` filters from a filter-major flat slice.
    fn filters<const F: usize, const K: usize, const C: usize, const Q: usize>(
        values: &[i8],
        scales: [f32; Q],
        zero_points: [i8; Q],
    ) -> Tensor4D<i8, F, 1, K, C, Q> {
        Tensor4D::new(
            core::array::from_fn(|f| {
                Buffer2D::from_fn(|_, m| core::array::from_fn(|c| values[f * K * C + m * C + c]))
            }),
            scales,
            zero_points,
        )
    }

    fn biases<const F: usize, const Q: usize>(values: &[i32]) -> Tensor2D<i32, F, 1, Q> {
        Tensor2D::new(Buffer2D::from_fn(|f, _| values[f]), [0.; Q], [0; Q])
    }

    /// Identity requant: scale_x · scale_w = scale_out, all zero points zero.
    #[test]
    fn accumulator_dot_product() {
        // x = [1 2 3 4 5], w = [1 0 -1]: every valid window gives -2
        let output = conv_1d::<_, 5, 1, 3, 1, 1, 3>(
            input::<5, 1>(&[1, 2, 3, 4, 5], 0.5, 0),
            &filters::<1, 3, 1, 1>(&[1, 0, -1], [0.5], [0]),
            &biases::<1, 1>(&[0]),
            [0.25],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Valid,
                stride: 1,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], -2);
        assert_eq!(output.buffer[0][(0, 1)][0], -2);
        assert_eq!(output.buffer[0][(0, 2)][0], -2);
    }

    /// Full pipeline with non-trivial quantization, per-channel scales,
    /// biases, and ties-to-even rounding cases (raw · m ends in .5).
    #[test]
    fn requant_per_channel_with_ties() {
        // x - zp_x = [5 15 25 35]
        // f0: w = [1 2], zp 0  -> acc = [35 65 95], + 7 -> raw = [42 72 102],
        //     m = 0.5 -> [21 36 51]
        // f1: w = [-1 1], zp 3 -> acc = [-50 -110 -170], - 8 -> raw = [-58 -118 -178],
        //     m = 0.25: raw · m = [-14.5 -29.5 -44.5] -> ties_even -> [-14 -30 -44]
        // out = round(raw · m) + zp_out, zp_out = -3
        let output = conv_1d::<_, 4, 1, 2, 2, 2, 3>(
            input::<4, 1>(&[10, 20, 30, 40], 0.5, 5),
            &filters::<2, 2, 1, 2>(&[1, 2, -1, 1], [0.1, 0.05], [0, 3]),
            &biases::<2, 2>(&[7, -8]),
            [0.1],
            [-3],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Valid,
                stride: 1,
            },
        );
        let expected = [[18i8, -17], [33, -33], [48, -47]];
        for (t, pair) in expected.iter().enumerate() {
            assert_eq!(output.buffer[0][(0, t)][0], pair[0], "t={t}, f=0");
            assert_eq!(output.buffer[0][(0, t)][1], pair[1], "t={t}, f=1");
        }
    }

    /// Fused ReLU clamps at the output zero point, in quantized coordinates.
    #[test]
    fn fused_relu_clamps_at_output_zero_point() {
        let output = conv_1d::<_, 4, 1, 2, 2, 2, 3>(
            input::<4, 1>(&[10, 20, 30, 40], 0.5, 5),
            &filters::<2, 2, 1, 2>(&[1, 2, -1, 1], [0.1, 0.05], [0, 3]),
            &biases::<2, 2>(&[7, -8]),
            [0.1],
            [-3],
            Conv1DOptions {
                fused_activation: FusedActivation::Relu,
                padding: TensorViewPadding::Valid,
                stride: 1,
            },
        );
        for t in 0..3 {
            assert_eq!(output.buffer[0][(0, t)][0], (18 + 15 * t) as i8);
            assert_eq!(output.buffer[0][(0, t)][1], -3, "clipped to zp_out");
        }
    }

    #[test]
    fn stride_two() {
        // Windows [x0 x1] and [x2 x3] with w = [1 1]
        let output = conv_1d::<_, 5, 1, 2, 1, 1, 2>(
            input::<5, 1>(&[0, 1, 2, 3, 4], 1., 0),
            &filters::<1, 2, 1, 1>(&[1, 1], [1.], [0]),
            &biases::<1, 1>(&[0]),
            [1.],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Valid,
                stride: 2,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 1);
        assert_eq!(output.buffer[0][(0, 1)][0], 5);
    }

    /// Same padding: pad_left = 1, both windows lose one out-of-range tap.
    #[test]
    fn same_padding_pads_left() {
        // T = 2 < k = 3: out = ceil(2/1) = 2, pad_total = 2, pad_left = 1
        let output = conv_1d::<_, 2, 1, 3, 1, 1, 2>(
            input::<2, 1>(&[1, 2], 1., 0),
            &filters::<1, 3, 1, 1>(&[1, 1, 1], [1.], [0]),
            &biases::<1, 1>(&[0]),
            [1.],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Same,
                stride: 1,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 3);
        assert_eq!(output.buffer[0][(0, 1)][0], 3);
    }

    /// Same padding with an odd total puts the extra tap on the right;
    /// out-of-range positions are neutral regardless of zp_x.
    #[test]
    fn same_padding_extra_tap_on_the_right() {
        // T = 4, k = 3, stride = 2: out = 2, pad_total = 1, pad_left = 0
        let output = conv_1d::<_, 4, 1, 3, 1, 1, 2>(
            input::<4, 1>(&[1, 2, 3, 4], 1., 0),
            &filters::<1, 3, 1, 1>(&[1, 1, 1], [1.], [0]),
            &biases::<1, 1>(&[0]),
            [1.],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Same,
                stride: 2,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 1 + 2 + 3);
        assert_eq!(output.buffer[0][(0, 1)][0], 3 + 4);
        // Same input shifted through zp_x = 5: identical results,
        // because (zp_x - zp_x) · w contributes nothing
        let output = conv_1d::<_, 4, 1, 3, 1, 1, 2>(
            input::<4, 1>(&[6, 7, 8, 9], 1., 5),
            &filters::<1, 3, 1, 1>(&[1, 1, 1], [1.], [0]),
            &biases::<1, 1>(&[0]),
            [1.],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Same,
                stride: 2,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 6);
        assert_eq!(output.buffer[0][(0, 1)][0], 7);
    }

    /// Eight channels: the dot product spans the full channel axis.
    #[test]
    fn eight_channels() {
        // w = [1 0 0 0 0 0 0 -1]: out(t) = x[t,0] - x[t,7]
        let mut values = [0i8; 3 * 8];
        for t in 0..3 {
            for c in 0..8 {
                values[t * 8 + c] = if c == 7 { (1 + t) as i8 } else { (1 + c) as i8 };
            }
        }
        let output = conv_1d::<_, 3, 8, 1, 1, 1, 3>(
            input::<3, 8>(&values, 1., 0),
            &filters::<1, 1, 8, 1>(&[1, 0, 0, 0, 0, 0, 0, -1], [1.], [0]),
            &biases::<1, 1>(&[0]),
            [1.],
            [0],
            Conv1DOptions {
                fused_activation: FusedActivation::None,
                padding: TensorViewPadding::Valid,
                stride: 1,
            },
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 0);
        assert_eq!(output.buffer[0][(0, 1)][0], -1);
        assert_eq!(output.buffer[0][(0, 2)][0], -2);
    }

    #[test]
    fn geometry_helpers() {
        use TensorViewPadding::{Same, Valid};
        assert_eq!(output_len(5, 3, 1, Valid), Some(3));
        assert_eq!(output_len(5, 3, 2, Valid), Some(2));
        assert_eq!(output_len(2, 3, 1, Valid), None);
        assert_eq!(output_len(2, 3, 1, Same), Some(2));
        assert_eq!(output_len(4, 3, 2, Same), Some(2));
        assert_eq!(same_pad_left(2, 1, 3, 2), 1);
        assert_eq!(same_pad_left(2, 2, 3, 4), 0);
        assert_eq!(same_pad_left(5, 1, 3, 5), 1);
    }

    /// The ties-even helper matches the reference semantics on the tricky
    /// values (exact halves round to the even neighbor, negatives included).
    #[test]
    fn ties_even_rounding() {
        assert_eq!(round_ties_even(0.5), 0.0);
        assert_eq!(round_ties_even(1.5), 2.0);
        assert_eq!(round_ties_even(2.5), 2.0);
        assert_eq!(round_ties_even(3.5), 4.0);
        assert_eq!(round_ties_even(-0.5), -0.0);
        assert_eq!(round_ties_even(-1.5), -2.0);
        assert_eq!(round_ties_even(-2.5), -2.0);
        assert_eq!(round_ties_even(2.4), 2.0);
        assert_eq!(round_ties_even(2.6), 3.0);
        assert_eq!(round_ties_even(-2.6), -3.0);
        assert_eq!(round_ties_even(f32::INFINITY), f32::INFINITY);
        assert!(round_ties_even(f32::NAN).is_nan());
    }
}
