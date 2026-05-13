use std::any::type_name;
use std::mem::size_of;

use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{DMatrix, Dyn};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;
use syn::{parse_str, Type};

use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
use crate::quantize::TokenQuantized;
use crate::tflite_flatbuffers::tflite::{Buffer, Padding, Tensor};

/// Represents the tokenized version of the `TensorViewPadding`.
#[derive(Copy, Clone)]
pub(crate) enum TokenTensorViewPadding {
    Same,
    Valid,
}

/// Represents the tokenized version of the `Tensor2D`.
#[derive(Debug)]
pub(crate) struct TokenTensor2D<T: TokenQuantized> {
    pub(crate) buffer: TokenBuffer2D<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scale: Vec<f32>,
    pub(crate) zero_point: Vec<T>,
}

/// Represents the tokenized version of the `Tensor4D`.
#[derive(Debug)]
pub(crate) struct TokenTensor4D<T: TokenQuantized> {
    pub(crate) buffer: TokenBuffer4D<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scale: Vec<f32>,
    pub(crate) zero_point: Vec<T>,
}

impl ToTokens for TokenTensorViewPadding {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::Same => quote!(microflow::tensor::TensorViewPadding::Same),
            Self::Valid => quote!(microflow::tensor::TensorViewPadding::Valid),
        }
        .to_tokens(tokens);
    }
}

impl From<Padding> for TokenTensorViewPadding {
    fn from(padding: Padding) -> Self {
        match padding {
            Padding::SAME => Self::Same,
            Padding::VALID => Self::Valid,
            _ => unreachable!(),
        }
    }
}

impl<T: TokenQuantized> TokenTensor2D<T> {
    pub fn zeroed_tensor(shape: (usize, usize)) -> Self {
        let shape = vec![shape.0, shape.1];
        Self {
            buffer: TokenBuffer2D {
                0: Option::Some(DMatrix::from_element_generic(
                    Dyn(shape[0]),
                    Dyn(shape[1]),
                    T::zeroed(),
                )),
            },
            shape,
            scale: vec![0f32],
            zero_point: vec![T::from_superset_unchecked(&0f32)],
        }
    }
    /// Builds a [`TokenTensor2D`] from an empty [`Tensor`].
    ///
    /// # Arguments
    /// * `tensor` - The empty model tensor as a [`Tensor`]
    ///
    pub fn from_empty_tensor(tensor: Tensor) -> Self {
        let mut shape: Vec<_> = tensor.shape().unwrap().iter().map(|e| e as usize).collect();
        if shape.len() == 1 {
            shape.insert(0, 1);
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

    /// Builds a [`TokenTensor2D`] from a [`Tensor`] with a buffer.
    ///
    /// # Arguments
    /// * `tensor` - The model tensor as a [`Tensor`]
    /// * `buffer` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    ///
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
        token_tensor.shape.swap(0, 1);
        token_tensor
    }

    /// Returns the tokens of the [`Self`] type.
    pub fn type_tokens(&self) -> TokenStream2 {
        let ty = parse_str::<Type>(type_name::<T>()).unwrap();
        let shape = &self.shape;
        let quants = self.scale.len();
        quote!(microflow::tensor::Tensor2D<#ty, #(#shape),*, #quants>)
    }
}

impl<T: TokenQuantized> ToTokens for TokenTensor2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let buffer = &self.buffer;
        let scale = &self.scale;
        let zero_point = &self.zero_point;

        let ts = quote! {
            microflow::tensor::Tensor2D::new(
                #buffer,
                [#(#scale),*],
                [#(#zero_point),*]
            )
        };
        ts.to_tokens(tokens);
    }
}

impl<T: TokenQuantized> TokenTensor4D<T> {
    /// Builds a [`TokenTensor4D`] from an empty [`Tensor`].
    ///
    /// # Arguments
    /// * `tensor` - The empty model tensor as a [`Tensor`]
    ///
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

    /// Builds a [`TokenTensor4D`] from a [`Tensor`] with a buffer.
    ///
    /// # Arguments
    /// * `tensor` - The model tensor as a [`Tensor`]
    /// * `buffer` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    ///
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

    /// Returns the tokens of the [`Self`] type.
    pub fn type_tokens(&self) -> TokenStream2 {
        let ty = parse_str::<Type>(type_name::<T>()).unwrap();
        let shape = &self.shape;
        let quants = self.scale.len();
        quote!(microflow::tensor::Tensor4D<#ty, #(#shape),*, #quants>)
    }
}

impl<T: TokenQuantized> ToTokens for TokenTensor4D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let buffer = &self.buffer;
        let scale = &self.scale;
        let zero_point = &self.zero_point;

        let ts = quote! {
            microflow::tensor::Tensor4D::new(
                #buffer,
                [#(#scale),*],
                [#(#zero_point),*]
            )
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn setup_2d() -> TokenTensor2D<i8> {
        TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                1, 2, 3;
                4, 5, 6
            ]),
            shape: vec![2, 3],
            scale: vec![0.7],
            zero_point: vec![8],
        }
    }

    fn setup_4d() -> TokenTensor4D<i8> {
        TokenTensor4D {
            buffer: TokenBuffer4D::from(vec![
                dmatrix![
                    vec![9,  10], vec![11, 12], vec![13, 14];
                    vec![15, 16], vec![17, 18], vec![19, 20]
                ],
                dmatrix![
                    vec![21, 22], vec![23, 24], vec![25, 26];
                    vec![27, 28], vec![29, 30], vec![31, 32]
                ],
            ]),
            shape: vec![2, 2, 3, 2],
            scale: vec![0.33, 0.34],
            zero_point: vec![35, 36],
        }
    }

    #[test]
    fn view_padding_to_tokens() {
        let padding = TokenTensorViewPadding::from(Padding::VALID);
        assert_eq!(
            padding.to_token_stream().to_string(),
            quote!(microflow::tensor::TensorViewPadding::Valid).to_string()
        );
    }

    #[test]
    fn tensor_2d_type_tokens() {
        let tensor = setup_2d();
        assert_eq!(
            tensor.type_tokens().to_string(),
            quote!(microflow::tensor::Tensor2D<i8, 2usize, 3usize, 1usize>).to_string(),
        )
    }

    #[test]
    fn tensor_2d_to_tokens() {
        let tensor = setup_2d();
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
    fn tensor_4d_type_tokens() {
        let tensor = setup_4d();
        assert_eq!(
            tensor.type_tokens().to_string(),
            quote!(microflow::tensor::Tensor4D<i8, 2usize, 2usize, 3usize, 2usize, 2usize>)
                .to_string(),
        )
    }

    #[test]
    fn tensor_4d_to_tokens() {
        let tensor = setup_4d();
        let buffer = &tensor.buffer;
        assert_eq!(
            tensor.to_token_stream().to_string(),
            quote! {
                microflow::tensor::Tensor4D::new(
                    #buffer,
                    [0.33f32, 0.34f32],
                    [35i8, 36i8]
                )
            }
            .to_string()
        );
    }
}
