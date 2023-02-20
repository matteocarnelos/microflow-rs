use std::mem::size_of;

use byterepr::ByteReprNum;
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{DMatrix, Scalar};
use num_traits::Zero;
use simba::scalar::SupersetOf;

use crate::tflite_flatbuffers::tflite::{Buffer, Tensor};
use crate::{quote, ToTokens, TokenStream2};

#[derive(Debug)]
pub struct ParsedTensor<T> {
    pub(crate) matrix: DMatrix<T>,
    pub(crate) scale: f32,
    pub(crate) zero_point: T,
}

impl<T> ParsedTensor<T>
where
    T: Scalar + ByteReprNum,
    i64: SupersetOf<T>,
{
    pub fn new_empty(tensor: Tensor) -> Self
    where
        T: Zero,
    {
        let shape: Vec<usize> = tensor.shape().unwrap().iter().map(|e| e as usize).collect();
        Self {
            matrix: DMatrix::zeros(shape[0], shape[1]),
            scale: tensor.quantization().unwrap().scale().unwrap().get(0),
            zero_point: i64::to_subset_unchecked(
                &tensor.quantization().unwrap().zero_point().unwrap().get(0),
            ),
        }
    }

    pub fn new_with_data(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Self {
        let mut shape: Vec<usize> = tensor.shape().unwrap().iter().map(|e| e as usize).collect();
        if shape.len() == 1 {
            shape.insert(0, 1);
        }
        let data = buffers
            .get(tensor.buffer() as usize)
            .data()
            .unwrap()
            .bytes()
            .chunks(size_of::<T>())
            .map(|e| T::from_le_bytes(e))
            .collect();
        Self {
            matrix: DMatrix::from_vec(shape[1], shape[0], data),
            scale: tensor.quantization().unwrap().scale().unwrap().get(0),
            zero_point: i64::to_subset_unchecked(
                &tensor.quantization().unwrap().zero_point().unwrap().get(0),
            ),
        }
    }
}

impl<T> ToTokens for ParsedTensor<T>
where
    T: ToTokens,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let mut matrix = TokenStream2::new();
        for row in self.matrix.row_iter() {
            let iter = row.iter();
            let t = quote! { #(#iter),*; };
            t.to_tokens(&mut matrix);
        }
        let scale = self.scale;
        let zero_point = &self.zero_point;
        let tensor = quote! {
            microflow::tensor::QuantizedTensor::new(
                nalgebra::matrix![#matrix],
                #scale,
                #zero_point
            )
        };
        tensor.to_tokens(tokens);
    }
}
