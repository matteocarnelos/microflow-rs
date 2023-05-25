extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, ItemStruct};

use crate::tflite_flatbuffers::tflite::TensorType;
use ops::*;
use structmeta::StructMeta;
use syn::{LitInt, LitStr};
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod activation;
mod buffer;
mod ops;
mod quantize;
mod tensor;
#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
    capacity: Option<LitInt>,
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);

    let buf = fs::read(args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });

    let capacity = args.capacity.map(|x| {
        x.base10_parse::<usize>().unwrap_or_else(|_| {
            abort_call_site!("invalid value for parameter `capacity`: {}", x.to_string());
        })
    });

    let ident = item.ident;

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    let input_signature = match input_shape.len() {
        1 => quote!(microflow::buffer::Buffer2D<f32, 0, #(#input_shape),*>),
        2 => quote!(microflow::buffer::Buffer2D<f32, #(#input_shape),*>),
        4 => quote!(microflow::buffer::Buffer4D<f32, #(#input_shape),*>),
        _ => unimplemented!(),
    };

    let input_scale: Vec<_> = input
        .quantization()
        .unwrap()
        .scale()
        .unwrap()
        .iter()
        .map(|e| e.to_token_stream())
        .collect();
    let input_zero_point: Vec<_> = match input.type_() {
        TensorType::INT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as i8).to_token_stream())
            .collect(),
        TensorType::UINT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as u8).to_token_stream())
            .collect(),
        _ => unimplemented!(),
    };
    let input_quantization = match input_shape.len() {
        1 => {
            quote!(microflow::tensor::Tensor2D::quantize(input, #(#input_scale)*, #(#input_zero_point)*))
        }
        2 => {
            quote!(microflow::tensor::Tensor2D::quantize(input, #(#input_scale)*, #(#input_zero_point)*))
        }
        4 => {
            quote!(microflow::tensor::Tensor4D::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]))
        }
        _ => unimplemented!(),
    };

    let operators = subgraph.operators().unwrap();
    let mut layers = TokenStream2::new();
    for operator in operators {
        let layer: Box<dyn ToTokens> = match BuiltinOperator(
            model
                .operator_codes()
                .unwrap()
                .get(operator.opcode_index() as usize)
                .deprecated_builtin_code() as i32,
        ) {
            BuiltinOperator::FULLY_CONNECTED => {
                fully_connected::parse(operator, tensors, buffers, capacity)
            }
            BuiltinOperator::DEPTHWISE_CONV_2D => {
                depthwise_conv_2d::parse(operator, tensors, buffers)
            }
            BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors),
            unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
        };
        layer.to_tokens(&mut layers)
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    let output_signature = match output_shape.len() {
        1 => quote!(microflow::buffer::Buffer2D<f32, 0, #(#output_shape),*>),
        2 => quote!(microflow::buffer::Buffer2D<f32, #(#output_shape),*>),
        4 => quote!(microflow::buffer::Buffer4D<f32, #(#output_shape),*>),
        _ => unimplemented!(),
    };

    let tokens = quote! {
        struct #ident;
        impl #ident {
            pub fn predict(input: #input_signature) -> #output_signature {
                let output = #input_quantization;
                #layers
                output.dequantize()
            }
        }
    };
    tokens.into()
}
