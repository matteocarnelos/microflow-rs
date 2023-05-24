use byterepr::ByteReprNum;
use nalgebra::Scalar;
use quote::ToTokens;
use simba::scalar::SubsetOf;

pub(crate) trait TokenQuantized:
    Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<i64>
{
}
impl<T: Scalar + ByteReprNum + ToTokens + SubsetOf<i32> + SubsetOf<i64>> TokenQuantized for T {}
