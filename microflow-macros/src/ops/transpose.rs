use std::mem::size_of;

use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};

/// Parsed TFLite `Transpose`: input tensor + `perm` (INT32 buffer).
pub(crate) struct TokenTranspose {
    pub(crate) perm: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> Box<dyn ToTokens> {
    Box::new(TokenTranspose::new(operator, tensors, buffers))
}

fn read_i32_perm(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Vec<i32> {
    let data = buffers
        .get(tensor.buffer() as usize)
        .data()
        .unwrap_or_else(|| abort_call_site!("transpose: missing perm buffer"))
        .bytes();
    data.chunks_exact(size_of::<i32>())
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl TokenTranspose {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator
            .inputs()
            .unwrap_or_else(|| abort_call_site!("transpose: no inputs"));
        if inputs.len() < 2 {
            abort_call_site!("transpose requires two inputs (data, perm)");
        }
        let perm_tensor = tensors.get(inputs.get(1) as usize);
        if perm_tensor.type_() != TensorType::INT32 {
            abort_call_site!("transpose perm tensor must be INT32");
        }
        let perm_i32 = read_i32_perm(perm_tensor, buffers);
        let perm: Vec<usize> = perm_i32.iter().map(|&p| p as usize).collect();

        let mut output_shape: Vec<_> = tensors
            .get(operator.outputs().unwrap().get(0) as usize)
            .shape()
            .unwrap()
            .iter()
            .map(|e| e as usize)
            .collect();
        if output_shape.len() == 1 {
            output_shape.insert(0, 1);
        }

        let rank = output_shape.len();
        if perm.len() != rank {
            abort_call_site!("transpose perm length must match tensor rank");
        }

        Self { perm, output_shape }
    }
}

impl ToTokens for TokenTranspose {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let perm = &self.perm;
        let output_shape = &self.output_shape;
        let rank = output_shape.len();

        let ts = match rank {
            2 => quote! {
                let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::transpose_2d(input, [#(#perm),*]);
            },
            4 => quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                    microflow::ops::transpose_4d(input, [#(#perm),*]);
            },
            _ => abort_call_site!("transpose: only rank 2 and 4 are supported"),
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn token_transpose_to_tokens_2d() {
        let layer = TokenTranspose {
            perm: vec![1, 0],
            output_shape: vec![3, 2],
        };
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 3usize, 2usize, 1usize> =
                    microflow::ops::transpose_2d(input, [1usize, 0usize]);
            }
            .to_string()
        );
    }

    #[test]
    fn token_transpose_to_tokens_4d() {
        let layer = TokenTranspose {
            perm: vec![0, 1, 3, 2],
            output_shape: vec![1, 2, 2, 3],
        };
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 2usize, 3usize, 1usize> =
                    microflow::ops::transpose_4d(input, [0usize, 1usize, 3usize, 2usize]);
            }
            .to_string()
        );
    }
}
