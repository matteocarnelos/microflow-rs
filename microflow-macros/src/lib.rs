extern crate proc_macro;

use proc_macro::TokenStream;
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;
use tflite_flatbuffers::tflite;
use tflite_flatbuffers::tflite::BuiltinOperator;

mod layers;
#[path = "../target/flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
mod tflite_flatbuffers;

// TODO: Support more quantization types (not i8 only)

#[proc_macro_attribute]
pub fn model(input: TokenStream, _item: TokenStream) -> TokenStream {
    let path = parse_macro_input!(input as syn::LitStr).value();
    let buf = fs::read(path).unwrap();
    let model = tflite::root_as_model(&buf).unwrap();
    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let input_shape: Vec<usize> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    let input_scale = input.quantization().unwrap().scale().unwrap().get(0);
    let input_zero_point = input.quantization().unwrap().zero_point().unwrap().get(0) as i8;

    let buffers = model.buffers().unwrap();
    let operators = subgraph.operators().unwrap();
    let operator_codes = model.operator_codes().unwrap();
    let mut layers: Vec<TokenStream2> = Vec::new();
    for operator in operators {
        layers.push(
            match operator_codes
                .get(operator.opcode_index() as usize)
                .builtin_code()
            {
                BuiltinOperator::FULLY_CONNECTED => {
                    layers::fully_connected(tensors, operator, buffers)
                }
                _ => unimplemented!(),
            },
        );
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let output_shape: Vec<usize> = output.shape().unwrap().iter().map(|e| e as usize).collect();

    let tokens = quote! {
        struct Model;
        impl Model {
            pub fn evaluate(input: nalgebra::SMatrix<f32, #(#input_shape),*>) -> nalgebra::SMatrix<f32, #(#output_shape),*> {
                let output = microflow::tensor::QuantizedTensor::quantize(input, #input_scale, #input_zero_point);
                #(#layers)*
                output.dequantize()
            }
        }
    };
    tokens.into()
}
