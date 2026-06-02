use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};

enum TokenLogistic<T: TokenQuantized> {
    D2(TokenTensor2D<T>),
    D4(TokenTensor4D<T>),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenLogistic::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenLogistic::<u8>::new(operator, tensors)),
        input_type => abort_call_site!(
            "Logistic supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenLogistic<T> {
    fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output = tensors.get(operator.outputs().unwrap().get(0) as usize);
        let rank = output.shape().unwrap().len();
        match rank {
            2 => Self::D2(TokenTensor2D::from_empty_tensor(output)),
            4 => Self::D4(TokenTensor4D::from_empty_tensor(output)),
            _ => abort_call_site!("Logistic supports only rank-2 and rank-4 tensors"),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenLogistic<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::D2(output) => {
                let output_shape = &output.shape;
                let output_scale = &output.scale;
                let output_zero_point = &output.zero_point;
                quote! {
                    let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                        microflow::ops::logistic_2d(input, [#(#output_scale),*], [#(#output_zero_point),*]);
                }
                .to_tokens(tokens);
            }
            Self::D4(output) => {
                let output_shape = &output.shape;
                let output_scale = &output.scale;
                let output_zero_point = &output.zero_point;
                quote! {
                    let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                        microflow::ops::logistic_4d(input, [#(#output_scale),*], [#(#output_zero_point),*]);
                }
                .to_tokens(tokens);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_2d_to_tokens() {
        let output = TokenLogistic::D2(TokenTensor2D {
            buffer: crate::buffer::TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: vec![0.2],
            zero_point: vec![3],
        });
        assert_eq!(
            output.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 1usize, 2usize, 1usize> =
                    microflow::ops::logistic_2d(input, [0.2f32], [3i32]);
            }
            .to_string()
        );
    }
}
