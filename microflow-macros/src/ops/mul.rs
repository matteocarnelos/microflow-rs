use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::Ident;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote, ToTokens};

enum TokenMul<T: TokenQuantized> {
    D2(TokenMul2D<T>),
    D4(TokenMul4D<T>),
}

pub(crate) struct TokenMul2D<T: TokenQuantized> {
    factor: TokenFactor2D<T>,
    output: TokenTensor2D<T>,
    fused_activation: TokenFusedActivation,
    index: usize,
}

pub(crate) struct TokenMul4D<T: TokenQuantized> {
    factor: TokenFactor4D<T>,
    output: TokenTensor4D<T>,
    fused_activation: TokenFusedActivation,
    index: usize,
}

enum TokenFactor2D<T: TokenQuantized> {
    Constant(TokenTensor2D<T>),
    Dynamic(Ident),
}

enum TokenFactor4D<T: TokenQuantized> {
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
        TensorType::INT8 => Box::new(TokenMul::<i8>::new(operator, tensors, buffers, index)),
        TensorType::UINT8 => Box::new(TokenMul::<u8>::new(operator, tensors, buffers, index)),
        input_type => abort_call_site!(
            "Mul supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenMul<T> {
    fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        if inputs.len() != 2 {
            abort_call_site!("Mul expects exactly two input tensors");
        }
        let lhs = tensors.get(inputs.get(0) as usize);
        let rhs = tensors.get(inputs.get(1) as usize);
        let options = operator.builtin_options_as_mul_options().unwrap();
        let rhs_buffer = buffers.get(rhs.buffer() as usize);
        let output = tensors.get(operator.outputs().unwrap().get(0) as usize);
        let rank = lhs.shape().unwrap().len();
        match rank {
            2 => Self::D2(TokenMul2D {
                factor: if rhs_buffer.data().is_some() {
                    TokenFactor2D::Constant(TokenTensor2D::from_buffered_tensor(rhs, buffers))
                } else {
                    TokenFactor2D::Dynamic(format_ident!("tensor_{}", inputs.get(1) as usize))
                },
                output: TokenTensor2D::from_empty_tensor(output),
                fused_activation: options.fused_activation_function().into(),
                index,
            }),
            4 => Self::D4(TokenMul4D {
                factor: if rhs_buffer.data().is_some() {
                    TokenFactor4D::Constant(TokenTensor4D::from_buffered_tensor(rhs, buffers))
                } else {
                    TokenFactor4D::Dynamic(format_ident!("tensor_{}", inputs.get(1) as usize))
                },
                output: TokenTensor4D::from_empty_tensor(output),
                fused_activation: options.fused_activation_function().into(),
                index,
            }),
            _ => abort_call_site!("Mul supports only rank-2 and rank-4 tensors"),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenMul<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        match self {
            Self::D2(layer) => layer.to_tokens(tokens),
            Self::D4(layer) => layer.to_tokens(tokens),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenMul2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let factor_ident = format_ident!("factor_{}", self.index);
        let (factor_declaration, factor_ref) = match &self.factor {
            TokenFactor2D::Constant(factor) => {
                let factor_type = factor.type_tokens();
                (
                    quote!(const #factor_ident: #factor_type = #factor;),
                    quote!(&#factor_ident),
                )
            }
            TokenFactor2D::Dynamic(ident) => (quote!(), quote!(&#ident)),
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        quote! {
            #factor_declaration
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> = microflow::ops::mul_2d(
                input,
                #factor_ref,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::MulOptions { fused_activation: #fused_activation },
            );
        }
        .to_tokens(tokens);
    }
}

impl<T: TokenQuantized> ToTokens for TokenMul4D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let factor_ident = format_ident!("factor_{}", self.index);
        let (factor_declaration, factor_ref) = match &self.factor {
            TokenFactor4D::Constant(factor) => {
                let factor_type = factor.type_tokens();
                (
                    quote!(const #factor_ident: #factor_type = #factor;),
                    quote!(&#factor_ident),
                )
            }
            TokenFactor4D::Dynamic(ident) => (quote!(), quote!(&#ident)),
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        quote! {
            #factor_declaration
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> = microflow::ops::mul_4d(
                input,
                #factor_ref,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::MulOptions { fused_activation: #fused_activation },
            );
        }
        .to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_2d_to_tokens() {
        let layer = TokenMul2D {
            factor: TokenFactor2D::Constant(TokenTensor2D {
                buffer: crate::buffer::TokenBuffer2D::from(nalgebra::dmatrix![1, 2]),
                shape: vec![1, 2],
                scale: vec![0.2],
                zero_point: vec![3],
            }),
            output: TokenTensor2D {
                buffer: crate::buffer::TokenBuffer2D::new(),
                shape: vec![1, 2],
                scale: vec![0.4],
                zero_point: vec![5],
            },
            fused_activation: TokenFusedActivation::None,
            index: 4,
        };
        let factor = match &layer.factor {
            TokenFactor2D::Constant(factor) => factor,
            TokenFactor2D::Dynamic(_) => panic!("expected constant factor"),
        };
        let fused_activation = layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const factor_4: microflow::tensor::Tensor2D<i32, 1usize, 2usize, 1usize> = #factor;
                let input: microflow::tensor::Tensor2D<_, 1usize, 2usize, 1usize> = microflow::ops::mul_2d(
                    input,
                    &factor_4,
                    [0.4f32],
                    [5i32],
                    microflow::ops::MulOptions { fused_activation: #fused_activation },
                );
            }
            .to_string()
        );
    }
}
