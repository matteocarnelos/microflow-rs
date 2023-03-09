use std::mem::size_of;

use byterepr::ByteReprNum;
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{DMatrix, Scalar};
use num_traits::Zero;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::matrix::TokenMatrix;
use crate::tflite_flatbuffers::tflite::{Buffer, Tensor};

#[derive(Debug)]
pub struct TokenTensor<T> {
    pub(crate) matrix: TokenMatrix<T>,
    pub(crate) scale: f32,
    pub(crate) zero_point: T,
}

impl<T> TokenTensor<T>
where
    T: Scalar + ByteReprNum,
    i64: SupersetOf<T>,
{
    pub fn new(matrix: TokenMatrix<T>, scale: f32, zero_point: T) -> Self {
        Self {
            matrix,
            scale,
            zero_point,
        }
    }

    pub fn from_buffered_tensor(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Self {
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
            matrix: DMatrix::from_vec(shape[1], shape[0], data).into(),
            scale: tensor.quantization().unwrap().scale().unwrap().get(0),
            zero_point: i64::to_subset_unchecked(
                &tensor.quantization().unwrap().zero_point().unwrap().get(0),
            ),
        }
    }
}

impl<T> From<Tensor<'_>> for TokenTensor<T>
where
    T: Scalar + Zero,
    i64: SupersetOf<T>,
{
    fn from(tensor: Tensor) -> Self {
        let shape: Vec<usize> = tensor.shape().unwrap().iter().map(|e| e as usize).collect();
        Self {
            matrix: DMatrix::zeros(shape[0], shape[1]).into(),
            scale: tensor.quantization().unwrap().scale().unwrap().get(0),
            zero_point: i64::to_subset_unchecked(
                &tensor.quantization().unwrap().zero_point().unwrap().get(0),
            ),
        }
    }
}

impl<T> ToTokens for TokenTensor<T>
where
    T: ToTokens,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let matrix = &self.matrix;
        let scale = &self.scale;
        let zero_point = &self.zero_point;

        let output = quote! {
            microflow::tensor::QuantizedTensor::new(
                #matrix,
                #scale,
                #zero_point
            )
        };
        output.to_tokens(tokens);
    }
}
