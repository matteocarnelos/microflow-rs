use byterepr::ByteReprNum;
use nalgebra::Scalar;
use quote::ToTokens;
use simba::scalar::SubsetOf;

/// Represents the trait to constrain a type to be quantized and tokenized.
pub(crate) trait TokenQuantized:
    Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>
{
}

impl<T: Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<f32> + SubsetOf<i64>>
    TokenQuantized for T
{
}
