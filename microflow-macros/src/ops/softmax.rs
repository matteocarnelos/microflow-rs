use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct Softmax<T: TokenQuantized> {
    pub(crate) input: TokenTensor2D<T>,
    pub(crate) output: TokenTensor2D<T>,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(Softmax::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(Softmax::<u8>::new(operator, tensors)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> Softmax<T> {
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor2D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { input, output }
    }
}

impl<T: TokenQuantized> ToTokens for Softmax<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let input_shape = &self.input.shape;
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let output = quote! {
            let input: microflow::tensor::Tensor2D<_, #(#input_shape),*, 1usize> = input.into();
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> = microflow::ops::softmax(input, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer2D;

    fn setup() -> Softmax<i8> {
        Softmax {
            input: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![2, 3],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![2, 3],
                scale: vec![0.3],
                zero_point: vec![4],
            },
        }
    }

    #[test]
    fn softmax_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 2usize, 3usize, 1usize> = input.into();
                let input: microflow::tensor::Tensor2D<_, 2usize, 3usize, 1usize> = microflow::ops::softmax(
                    input,
                    [0.3f32],
                    [4i8]
                );
            }
            .to_string()
        )
    }
}
