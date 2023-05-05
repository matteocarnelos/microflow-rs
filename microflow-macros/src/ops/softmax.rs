use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct Softmax {
    pub(crate) output: TokenTensor2D<i8>,
}

impl Softmax {
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output }
    }
}

impl ToTokens for Softmax {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let output = quote! {
            let output = microflow::ops::softmax(output.into(), #output_scale, #output_zero_point);
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer2D;

    #[test]
    fn softmax_to_tokens() {
        let layer = Softmax {
            output: TokenTensor2D {
                buffer: TokenBuffer2D::<i8>::new(),
                shape: vec![1, 2],
                scale: 0.3,
                zero_point: 4,
            },
        };
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output = microflow::ops::softmax(
                    output.into(),
                    0.3f32,
                    4i8
                );
            }
            .to_string()
        )
    }
}
