use std::mem::size_of;

use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
use crate::quantize::TokenQuantized;
use crate::tflite_flatbuffers::tflite::{Buffer, Padding, Tensor};

#[derive(Copy, Clone)]
pub(crate) enum TokenViewPadding {
    Same,
    Valid,
}

#[derive(Debug)]
pub(crate) struct TokenTensor2D<T: TokenQuantized> {
    pub(crate) buffer: TokenBuffer2D<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scale: Vec<f32>,
    pub(crate) zero_point: Vec<T>,
}

#[derive(Debug)]
pub(crate) struct TokenTensor4D<T: TokenQuantized> {
    pub(crate) buffer: TokenBuffer4D<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scale: Vec<f32>,
    pub(crate) zero_point: Vec<T>,
}

impl ToTokens for TokenViewPadding {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::Same => quote!(microflow::tensor::ViewPadding::Same),
            Self::Valid => quote!(microflow::tensor::ViewPadding::Valid),
        }
        .to_tokens(tokens);
    }
}

impl From<Padding> for TokenViewPadding {
    fn from(padding: Padding) -> Self {
        match padding {
            Padding::SAME => Self::Same,
            Padding::VALID => Self::Valid,
            _ => unreachable!(),
        }
    }
}

impl<T: TokenQuantized> TokenTensor2D<T> {
    pub fn from_empty_tensor(tensor: Tensor) -> Self {
        let mut shape: Vec<_> = tensor.shape().unwrap().iter().map(|e| e as usize).collect();
        if shape.len() == 1 {
            shape.insert(0, 1);
        }
        if shape.len() == 4 {
            shape = vec![1, shape[1] * shape[2] * shape[3]]
        }
        Self {
            buffer: TokenBuffer2D::new(),
            shape,
            scale: tensor
                .quantization()
                .unwrap()
                .scale()
                .unwrap()
                .iter()
                .collect(),
            zero_point: tensor
                .quantization()
                .unwrap()
                .zero_point()
                .unwrap()
                .iter()
                .map(|e| i64::to_subset_unchecked(&e))
                .collect(),
        }
    }

    pub fn from_buffered_tensor(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Self {
        let mut token_tensor = Self::from_empty_tensor(tensor);
        let matrix = DMatrix::from_iterator(
            token_tensor.shape[1],
            token_tensor.shape[0],
            buffers
                .get(tensor.buffer() as usize)
                .data()
                .unwrap()
                .bytes()
                .chunks_exact(size_of::<T>())
                .map(|e| T::from_le_bytes(e)),
        );
        token_tensor.buffer = TokenBuffer2D::from(matrix);
        token_tensor
    }
}

impl<T: TokenQuantized> ToTokens for TokenTensor2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let buffer = &self.buffer;
        let scale = &self.scale;
        let zero_point = &self.zero_point;
        let output = quote! {
            microflow::tensor::Tensor2D::new(
                #buffer,
                [#(#scale),*],
                [#(#zero_point),*]
            )
        };
        output.to_tokens(tokens);
    }
}

impl<T: TokenQuantized> TokenTensor4D<T> {
    pub fn from_empty_tensor(tensor: Tensor) -> Self {
        Self {
            buffer: TokenBuffer4D::new(),
            shape: tensor.shape().unwrap().iter().map(|e| e as usize).collect(),
            scale: tensor
                .quantization()
                .unwrap()
                .scale()
                .unwrap()
                .iter()
                .collect(),
            zero_point: tensor
                .quantization()
                .unwrap()
                .zero_point()
                .unwrap()
                .iter()
                .map(|e| i64::to_subset_unchecked(&e))
                .collect(),
        }
    }

    pub fn from_buffered_tensor(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Self {
        let mut t = Self::from_empty_tensor(tensor);
        let len = t.shape.iter().product::<usize>() * size_of::<T>();
        let data = buffers
            .get(tensor.buffer() as usize)
            .data()
            .unwrap()
            .bytes()
            .chunks_exact(len / t.shape[0])
            .map(|m| {
                DMatrix::from_row_iterator(
                    t.shape[1],
                    t.shape[2],
                    m.chunks_exact(len / (t.shape[0] * t.shape[1] * t.shape[2]))
                        .map(|v| {
                            v.chunks_exact(size_of::<T>())
                                .map(|e| T::from_le_bytes(e))
                                .collect::<Vec<_>>()
                        }),
                )
            })
            .collect::<Vec<_>>();
        t.buffer = TokenBuffer4D::from(data);
        t
    }
}

impl<T: TokenQuantized> ToTokens for TokenTensor4D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let buffer = &self.buffer;
        let scale = &self.scale;
        let zero_point = &self.zero_point;
        let output = quote! {
            microflow::tensor::Tensor4D::new(
                #buffer,
                [#(#scale),*],
                [#(#zero_point),*]
            )
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    #[test]
    fn view_padding_to_tokens() {
        let padding = TokenViewPadding::from(Padding::VALID);
        assert_eq!(
            padding.to_token_stream().to_string(),
            quote!(microflow::tensor::ViewPadding::Valid).to_string()
        );
    }

    #[test]
    fn tensor_2d_to_tokens() {
        let tensor = TokenTensor2D {
            buffer: TokenBuffer2D::<i8>::from(dmatrix![
                1, 2, 3;
                4, 5, 6
            ]),
            shape: vec![2, 3],
            scale: vec![0.7],
            zero_point: vec![8],
        };
        let buffer = &tensor.buffer;
        assert_eq!(
            tensor.to_token_stream().to_string(),
            quote! {
                microflow::tensor::Tensor2D::new(
                    #buffer,
                    [0.7f32],
                    [8i8]
                )
            }
            .to_string()
        );
    }

    #[test]
    fn tensor_4d_to_tokens() {
        let tensor = TokenTensor4D {
            buffer: TokenBuffer4D::<i8>::from(vec![
                dmatrix![
                    vec![1, 2], vec![3, 4],  vec![5,  6];
                    vec![7, 8], vec![9, 10], vec![11, 12]
                ],
                dmatrix![
                    vec![13, 14], vec![15, 16], vec![17, 18];
                    vec![19, 20], vec![21, 22], vec![23, 24]
                ],
            ]),
            shape: vec![2, 2, 3, 2],
            scale: vec![0.25, 0.26],
            zero_point: vec![27, 28],
        };
        let buffer = &tensor.buffer;
        assert_eq!(
            tensor.to_token_stream().to_string(),
            quote! {
                microflow::tensor::Tensor4D::new(
                    #buffer,
                    [0.25f32, 0.26f32],
                    [27i8, 28i8]
                )
            }
            .to_string()
        );
    }
}
