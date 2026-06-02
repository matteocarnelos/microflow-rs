use std::mem::size_of;

use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};

pub(crate) struct TokenPad<T: TokenQuantized> {
    output: TokenTensor4D<T>,
    paddings: [(usize, usize); 4],
    constant: T,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenPad::<i8>::new(operator, tensors, buffers)),
        TensorType::UINT8 => Box::new(TokenPad::<u8>::new(operator, tensors, buffers)),
        input_type => abort_call_site!(
            "Pad supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

fn read_i32_vector(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> Vec<i32> {
    let data = buffers
        .get(tensor.buffer() as usize)
        .data()
        .unwrap_or_else(|| abort_call_site!("Pad: missing buffer"))
        .bytes();
    data.chunks_exact(size_of::<i32>())
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_t_scalar<T: TokenQuantized>(tensor: Tensor, buffers: Vector<ForwardsUOffset<Buffer>>) -> T {
    let data = buffers
        .get(tensor.buffer() as usize)
        .data()
        .unwrap_or_else(|| abort_call_site!("PadV2: missing constant value buffer"))
        .bytes();
    T::from_le_bytes(
        data.get(0..size_of::<T>())
            .unwrap_or_else(|| abort_call_site!("PadV2: invalid constant value"))
            .try_into()
            .unwrap(),
    )
}

impl<T: TokenQuantized> TokenPad<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        if inputs.len() < 2 {
            abort_call_site!("Pad expects at least two inputs: data and paddings");
        }
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let paddings_raw = read_i32_vector(tensors.get(inputs.get(1) as usize), buffers);
        if paddings_raw.len() != 8 {
            abort_call_site!("Pad currently supports rank-4 tensors only");
        }
        let paddings = [
            (paddings_raw[0] as usize, paddings_raw[1] as usize),
            (paddings_raw[2] as usize, paddings_raw[3] as usize),
            (paddings_raw[4] as usize, paddings_raw[5] as usize),
            (paddings_raw[6] as usize, paddings_raw[7] as usize),
        ];
        let constant = if inputs.len() > 2 {
            read_t_scalar::<T>(tensors.get(inputs.get(2) as usize), buffers)
        } else {
            output.zero_point[0]
        };
        Self {
            output,
            paddings,
            constant,
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenPad<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let p00 = self.paddings[0].0;
        let p01 = self.paddings[0].1;
        let p10 = self.paddings[1].0;
        let p11 = self.paddings[1].1;
        let p20 = self.paddings[2].0;
        let p21 = self.paddings[2].1;
        let p30 = self.paddings[3].0;
        let p31 = self.paddings[3].1;
        let constant = self.constant;
        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::pad_4d(
                    input,
                    [(#p00, #p01), (#p10, #p11), (#p20, #p21), (#p30, #p31)],
                    #constant,
                );
        }
        .to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_tokens() {
        let layer = TokenPad {
            output: TokenTensor4D {
                buffer: crate::buffer::TokenBuffer4D::new(),
                shape: vec![1, 1, 4, 1],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            paddings: [(0, 0), (0, 0), (1, 1), (0, 0)],
            constant: -1,
        };
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 1usize, 4usize, 1usize, 1usize> =
                    microflow::ops::pad_4d(
                        input,
                        [(0usize, 0usize), (0usize, 0usize), (1usize, 1usize), (0usize, 0usize)],
                        -1i32,
                    );
            }
            .to_string()
        );
    }
}
