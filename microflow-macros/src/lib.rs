extern crate proc_macro;

use proc_macro::TokenStream;
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::parse_macro_input;

use tensor::ParsedTensor;
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod ops;
mod tensor;
#[path = "../target/flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
mod tflite_flatbuffers;

// TODO: Support more quantization types (not i8 only)

#[proc_macro_attribute]
pub fn model(input: TokenStream, _item: TokenStream) -> TokenStream {
    let path = parse_macro_input!(input as syn::LitStr).value();
    let buf = fs::read(path).unwrap();
    let model = root_as_model(&buf).unwrap();
    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let input: ParsedTensor<i8> = ParsedTensor::new_empty(input);

    let operators = subgraph.operators().unwrap();
    let operator_codes = model.operator_codes().unwrap();
    let mut layers = TokenStream2::new();
    for operator in operators {
        let layer = match operator_codes
            .get(operator.opcode_index() as usize)
            .builtin_code()
        {
            BuiltinOperator::FULLY_CONNECTED => {
                ops::FullyConnected::new(operator, tensors, buffers)
            }
            _ => unimplemented!(),
        };
        layer.to_tokens(&mut layers)
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let output: ParsedTensor<i8> = ParsedTensor::new_empty(output);

    let input_rows = input.matrix.shape().0;
    let input_columns = input.matrix.shape().1;
    let input_scale = input.scale;
    let input_zero_point = input.zero_point;

    let output_rows = output.matrix.shape().0;
    let output_columns = output.matrix.shape().1;

    let tokens = quote! {
        struct Model;
        impl Model {
            pub fn evaluate(input: nalgebra::SMatrix<f32, #input_rows, #input_columns>) -> nalgebra::SMatrix<f32, #output_rows, #output_columns> {
                let output = microflow::tensor::QuantizedTensor::quantize(input, #input_scale, #input_zero_point);
                #layers
                output.dequantize()
            }
        }
    };
    tokens.into()
}
