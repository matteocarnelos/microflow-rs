use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::Ident;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote, ToTokens};

pub(crate) struct TokenConcatenation<T: TokenQuantized> {
    tail: TokenConcatTail<T>,
    output: TokenTensor4D<T>,
    index: usize,
}

enum TokenConcatTail<T: TokenQuantized> {
    Constant(TokenTensor4D<T>),
    Dynamic(Ident),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenConcatenation::<i8>::new(operator, tensors, buffers, index)),
        TensorType::UINT8 => {
            Box::new(TokenConcatenation::<u8>::new(operator, tensors, buffers, index))
        }
        input_type => abort_call_site!(
            "Concatenation supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenConcatenation<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        if inputs.len() != 2 {
            abort_call_site!("Concatenation currently supports exactly two inputs");
        }
        let options = operator.builtin_options_as_concatenation_options().unwrap();
        if options.axis() != 3 && options.axis() != -1 {
            abort_call_site!("Concatenation currently supports only channel axis (3 or -1)");
        }
        let tail_tensor = tensors.get(inputs.get(1) as usize);
        let tail_buffer = buffers.get(tail_tensor.buffer() as usize);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self {
            tail: if tail_buffer.data().is_some() {
                TokenConcatTail::Constant(TokenTensor4D::from_buffered_tensor(tail_tensor, buffers))
            } else {
                TokenConcatTail::Dynamic(format_ident!("tensor_{}", inputs.get(1) as usize))
            },
            output,
            index,
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenConcatenation<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let tail_ident = format_ident!("concat_tail_{}", self.index);
        let (tail_declaration, tail_ref) = match &self.tail {
            TokenConcatTail::Constant(tail) => {
                let tail_type = tail.type_tokens();
                (
                    quote!(const #tail_ident: #tail_type = #tail;),
                    quote!(&#tail_ident),
                )
            }
            TokenConcatTail::Dynamic(ident) => (quote!(), quote!(&#ident)),
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        quote! {
            #tail_declaration
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::concatenation_4d(
                    input,
                    #tail_ref,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                );
        }
        .to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenation_to_tokens() {
        let layer = TokenConcatenation {
            tail: TokenConcatTail::Constant(TokenTensor4D {
                buffer: crate::buffer::TokenBuffer4D::from(vec![nalgebra::dmatrix![vec![1, 2]]]),
                shape: vec![1, 1, 1, 2],
                scale: vec![0.3],
                zero_point: vec![4],
            }),
            output: TokenTensor4D {
                buffer: crate::buffer::TokenBuffer4D::new(),
                shape: vec![1, 1, 1, 3],
                scale: vec![0.5],
                zero_point: vec![6],
            },
            index: 2,
        };
        let tail = match &layer.tail {
            TokenConcatTail::Constant(tail) => tail,
            TokenConcatTail::Dynamic(_) => panic!("expected constant tail"),
        };
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const concat_tail_2: microflow::tensor::Tensor4D<i32, 1usize, 1usize, 1usize, 2usize, 1usize> = #tail;
                let input: microflow::tensor::Tensor4D<_, 1usize, 1usize, 1usize, 3usize, 1usize> =
                    microflow::ops::concatenation_4d(
                        input,
                        &concat_tail_2,
                        [0.5f32],
                        [6i32],
                    );
            }
            .to_string()
        );
    }
}
