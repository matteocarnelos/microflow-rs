extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::parse_macro_input;

use structmeta::StructMeta;
use syn::{LitInt, LitStr};
use tensor::{TokenTensor2D, TokenTensor4D};
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod activation;
mod buffer;
mod ops;
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

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    let mut input_signature = TokenStream2::new();
    let mut input_quantization = TokenStream2::new();
    if input_shape.len() <= 2 {
        let input: TokenTensor2D<i8> = TokenTensor2D::from_empty_tensor(input);
        let input_shape = input.shape;
        let input_scale = input.scale;
        let input_zero_point = input.zero_point;
        quote!(microflow::buffer::Buffer2D<f32, #(#input_shape),*>).to_tokens(&mut input_signature);
        quote!(microflow::tensor::QuantizedTensor2D::quantize(input, #input_scale, #input_zero_point)).to_tokens(&mut input_quantization);
    } else if input_shape.len() == 4 {
        let input: TokenTensor4D<i8> = TokenTensor4D::from_empty_tensor(input);
        let input_shape = input.shape;
        let input_scale = input.scale;
        let input_zero_point = input.zero_point;
        quote!(microflow::buffer::Buffer4D<f32, #(#input_shape),*>).to_tokens(&mut input_signature);
        quote!(microflow::tensor::QuantizedTensor4D::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*])).to_tokens(&mut input_quantization);
    } else {
        unimplemented!()
    }

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
            BuiltinOperator::FULLY_CONNECTED => Box::new(ops::FullyConnected::new(
                operator, tensors, buffers, capacity,
            )),
            BuiltinOperator::DEPTHWISE_CONV_2D => {
                Box::new(ops::DepthwiseConv2D::new(operator, tensors, buffers))
            }
            BuiltinOperator::SOFTMAX => Box::new(ops::Softmax::new(operator, tensors)),
            unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
        };
        layer.to_tokens(&mut layers)
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    let mut output_signature = TokenStream2::new();
    if output_shape.len() <= 2 {
        let output: TokenTensor4D<i8> = TokenTensor4D::from_empty_tensor(output);
        let output_shape = output.shape;
        quote!(microflow::buffer::Buffer2D<f32, #(#output_shape),*>)
            .to_tokens(&mut output_signature);
    } else if output_shape.len() == 4 {
        let output: TokenTensor4D<i8> = TokenTensor4D::from_empty_tensor(output);
        let output_shape = output.shape;
        quote!(microflow::buffer::Buffer2D<f32, #(#output_shape),*>)
            .to_tokens(&mut output_signature);
    } else {
        unimplemented!()
    }

    let tokens = quote! {
        struct Model;
        impl Model {
            pub fn predict(input: #input_signature) -> #output_signature {
                let output = #input_quantization;
                #layers
                output.dequantize()
            }
        }
    };
    tokens.into()
}
