//! Backend selection helpers for optimized kernels.
//!
//! The default runtime path is pure Rust. Enabling the `cmsis-nn` feature
//! turns on the CMSIS-NN backend hooks for supported kernels.

use libm::{frexpf, roundf};

/// Returns true when the build enables the CMSIS-NN backend path.
#[inline(always)]
pub const fn use_cmsis_nn() -> bool {
    cfg!(feature = "cmsis-nn")
}

/// Converts a real multiplier into the CMSIS-NN quantized `(multiplier, shift)` pair.
///
/// This matches the fixed-point convention used by TFLite/CMSIS kernels where
/// `multiplier` is Q31 and `shift` is the base-2 exponent from `frexp`.
pub fn quantize_multiplier(real_multiplier: f32) -> (i32, i32) {
    if real_multiplier == 0.0 {
        return (0, 0);
    }
    let (significand, mut shift) = frexpf(real_multiplier);
    let mut q31 = roundf(significand * ((1u64 << 31) as f32)) as i64;
    if q31 == (1i64 << 31) {
        q31 /= 2;
        shift += 1;
    }
    (q31 as i32, shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_multiplier_zero() {
        assert_eq!(quantize_multiplier(0.0), (0, 0));
    }

    #[test]
    fn quantize_multiplier_nonzero() {
        let (m, s) = quantize_multiplier(0.506_896_56);
        assert_eq!(m > 0, true);
        assert_eq!(s <= 0, true);
    }
}

