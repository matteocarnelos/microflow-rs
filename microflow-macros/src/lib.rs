extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::parse_macro_input;

use layers::SUPPORTED_OPS;
use structmeta::StructMeta;
use syn::{LitInt, LitStr};
use tensor::TokenTensor;
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod layers;
mod matrix;
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
pub fn model(input: TokenStream, _item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as Args);
    let buf = fs::read(&args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });

    let operator_codes = model.operator_codes().unwrap();
    for operator_code in operator_codes {
        if !SUPPORTED_OPS.contains(&operator_code.builtin_code()) {
            abort_call_site!("unsupported operator: {:?}", operator_code.builtin_code());
        }
    }

    let capacity = args.capacity.map(|x| {
        x.base10_parse::<usize>().unwrap_or_else(|_| {
            abort_call_site!("invalid value for parameter `capacity`: {}", x.to_string());
        })
    });

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let input: TokenTensor<i8> = input.into();

    let operators = subgraph.operators().unwrap();
    let mut layers = TokenStream2::new();
    for operator in operators {
        let layer: Box<dyn ToTokens> = match operator_codes
            .get(operator.opcode_index() as usize)
            .builtin_code()
        {
            BuiltinOperator::FULLY_CONNECTED => Box::new(layers::FullyConnected::new(
                operator, tensors, buffers, capacity,
            )),
            _ => unreachable!(),
        };
        layer.to_tokens(&mut layers)
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let output: TokenTensor<i8> = output.into();

    let input_rows = input.matrix.shape().0;
    let input_columns = input.matrix.shape().1;
    let input_scale = input.scale;
    let input_zero_point = input.zero_point;

    let output_rows = output.matrix.shape().0;
    let output_columns = output.matrix.shape().1;

    let tokens = quote! {
        struct Model;
        impl Model {
            pub fn evaluate(input: microflow::tensor::Buffer2D<f32, #input_rows, #input_columns>) -> microflow::tensor::Buffer2D<f32, #output_rows, #output_columns> {
                let output = microflow::tensor::QuantizedTensor2D::quantize(input, #input_scale, #input_zero_point);
                #layers
                output.dequantize()
            }
        }
    };
    tokens.into()
}
