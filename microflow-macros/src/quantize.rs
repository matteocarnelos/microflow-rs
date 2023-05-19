use byterepr::ByteReprNum;
use nalgebra::Scalar;
use quote::ToTokens;
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub, SubsetOf, SupersetOf};

use num_traits::Zero;

pub trait TokenQuantized:
    Scalar
    + Zero
    + ClosedAdd
    + ClosedSub
    + ClosedMul
    + ByteReprNum
    + ToTokens
    + SubsetOf<i64>
    + SubsetOf<f32>
    + SupersetOf<usize>
{
}

impl<
        T: Scalar
            + Zero
            + ClosedAdd
            + ClosedSub
            + ClosedMul
            + ByteReprNum
            + ToTokens
            + SubsetOf<i64>
            + SubsetOf<f32>
            + SupersetOf<usize>,
    > TokenQuantized for T
{
}
