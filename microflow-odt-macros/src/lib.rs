//! [![crates.io](https://img.shields.io/crates/v/microflow-macros)](https://crates.io/crates/microflow-macros)
//! [![docs.rs](https://img.shields.io/docsrs/microflow-macros)](https://docs.rs/microflow-macros)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! Macro crate of the [MicroFlow](https://github.com/matteocarnelos/microflow-rs) inference engine, namely, the MicroFlow compiler.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quantize::TrainToTokens;
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, ExprArray, ItemStruct, LitBool};

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
    #[struct_meta(unnamed)]
    num_train_layers: LitInt,
    #[struct_meta(unnamed)]
    loss_function: LitStr,
    #[struct_meta(unnamed)]
    skip_last_layer_train: LitBool,
    #[struct_meta(unnamed)]
    back_norms: ExprArray,
    #[struct_meta(unnamed)]
    grad_norms: ExprArray,
}

/// The entry point of MicroFlow.
/// This attribute-like procedural macro can be placed on `structs` to implement the `predict()` and predict_train()
/// function based on the given model.
/// The macro takes as input:
/// - the path of the model, which must be in the TensorFlow Lite format (`.tflite`).
/// - the number of layers to train (from the end of the model),
/// - the loss function to use (either "mse" or "crossentropy"),
/// - whether to skip the training of the last layer,
/// - an array of float values representing the clipping norms for the backpropagation,
/// - an array of float values representing the clipping norms for the gradients.
/// differently from the inference, the training requires to store additional parameters for each layer to be trained 
/// and the accumulated gradients during the backpropagation.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);
    let mut item = item.clone();

    if args.back_norms.elems.len() != args.grad_norms.elems.len() {
        abort_call_site!("the clip values for the backpropagation and the gradient clipping must have the same size, found {}, {}",
             args.back_norms.elems.len(),
             args.grad_norms.elems.len());
    }
    for ele in args.back_norms.elems.iter() {
        if let syn::Expr::Lit(ele_lit) = ele {
            if let syn::Lit::Float(_) = &ele_lit.lit {
            } else {
                abort_call_site!("the arguments of the back_norms must be floats: {:?}", ele)
            }
        }
    }
    for ele in args.grad_norms.elems.iter() {
        if let syn::Expr::Lit(ele_lit) = ele {
            if let syn::Lit::Float(_) = &ele_lit.lit {
            } else {
                abort_call_site!("the arguments of the grad_norms must be floats: {:?}", ele)
            }
        }
    }
    let buf = fs::read(args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });
    let layers_to_train: usize = args.num_train_layers.base10_parse().unwrap();

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let mut input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    if input_shape.len() == 1 {
        input_shape.insert(0, 1);
    }
    let input_type = match input.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let input_tensor = match input_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let input_buffer = match input_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
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

    let operators = subgraph.operators().unwrap();
    let mut layers = TokenStream2::new();
    let mut layers_train = TokenStream2::new();
    let mut new_declarations = TokenStream2::new();
    let mut backward = TokenStream2::new();
    let mut update = TokenStream2::new();
    let mut layer_counter = 0;
    for (index, operator) in operators.iter().enumerate() {
        let layer_num = index as i32 - (operators.len() as i32 - layers_to_train as i32);
        if layer_num < 0 {
            let layer: Box<dyn ToTokens> = match BuiltinOperator(
                model
                    .operator_codes()
                    .unwrap()
                    .get(operator.opcode_index() as usize)
                    .deprecated_builtin_code() as i32,
            ) {
                BuiltinOperator::FULLY_CONNECTED => {
                    fully_connected::parse(operator, tensors, buffers, index)
                }
                BuiltinOperator::DEPTHWISE_CONV_2D => {
                    depthwise_conv_2d::parse(operator, tensors, buffers, index)
                }
                BuiltinOperator::CONV_2D => conv_2d::parse(operator, tensors, buffers, index),
                BuiltinOperator::AVERAGE_POOL_2D => average_pool_2d::parse(operator, tensors),
                BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors),
                BuiltinOperator::RESHAPE => Box::new(reshape::parse(operator, tensors)),
                unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
            };
            layer.to_tokens(&mut layers);
            layer.to_tokens(&mut layers_train);
        } else {
            let mut layer: Box<dyn TrainToTokens> = match BuiltinOperator(
                model
                    .operator_codes()
                    .unwrap()
                    .get(operator.opcode_index() as usize)
                    .deprecated_builtin_code() as i32,
            ) {
                BuiltinOperator::FULLY_CONNECTED => {
                    let back_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.back_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    let grad_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.grad_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    layer_counter += 1;
                    fully_connected::parse_indexed(
                        operator, tensors, buffers, index, layer_num, back_norm, grad_norm,
                    )
                }
                BuiltinOperator::DEPTHWISE_CONV_2D => {
                    let back_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.back_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    let grad_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.grad_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    layer_counter += 1;
                    depthwise_conv_2d::parse_indexed(
                        operator, tensors, buffers, index, layer_num, back_norm, grad_norm,
                    )
                }
                BuiltinOperator::CONV_2D => {
                    let back_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.back_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    let grad_norm = if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Float(lit_int),
                        ..
                    }) = &args.grad_norms.elems[layer_counter]
                    {
                        lit_int.base10_parse().unwrap()
                    } else {
                        abort_call_site!("error during norm parsing")
                    };
                    layer_counter += 1;
                    conv_2d::parse_indexed(
                        operator, tensors, buffers, index, layer_num, back_norm, grad_norm,
                    )
                }
                BuiltinOperator::AVERAGE_POOL_2D => {
                    layer_counter += 1;
                    average_pool_2d::parse_indexed(operator, tensors, layer_num)
                }
                BuiltinOperator::SOFTMAX => softmax::parse_indexed(operator, tensors, layer_num),
                BuiltinOperator::RESHAPE => reshape::parse_indexed(operator, tensors, layer_num),
                unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
            };
            layer.to_tokens(&mut layers);
            layer.switch_train();
            layer.to_tokens(&mut layers_train);
            if !(args.skip_last_layer_train.value && index >= operators.len() - 1) {
                layer.define_members(&mut new_declarations);
                layer.train_ops(&mut backward);
                layer.add_attrs(&mut item);
                layer.update_ops(&mut update);
            }
        }
    }

    let ident = &item.ident;
    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let mut output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    if output_shape.len() == 1 {
        output_shape.insert(0, 1);
    }
    let output_type = match output.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        _ => unimplemented!(),
    };
    let output_tensor = match output_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        _ => unimplemented!(),
    };
    let output_buffer = match output_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        _ => unimplemented!(),
    };

    let output_ident = format_ident!("input{}", layers_to_train - 1);
    let last_layer = operators.get(operators.len() - 1);
    let input_loss = if args.skip_last_layer_train.value {
        layers_to_train - 2
    } else {
        layers_to_train - 1
    };
    let loss_ident = match args.loss_function.value().as_str() {
        "crossentropy" => crossentropy::parse_indexed(last_layer, tensors, input_loss as i32),
        "mse" => mse::parse_indexed(last_layer, tensors, input_loss as i32),
        _ => unimplemented!(),
    };
    let loss_ident = loss_ident.into_token_stream();
    let ts = quote! {
        #item
        impl #ident {
            pub fn new() -> Self {
                #ident {
                    #new_declarations
                }
            }
            pub fn predict(&self, input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner(input).dequantize()
            }

            pub fn predict_quantized(&self, input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                //print!("dequantized: {}",input.dequantize()[0].map(|el|el[0]));
                self.predict_inner(input).dequantize()
            }

            fn predict_inner(&self, input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers
                input
            }
            pub fn predict_train(&mut self, input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>,output_gt : &microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize>,learning_rate: f32) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner_train(input, output_gt, learning_rate).dequantize()
            }

            pub fn predict_quantized_train(&mut self, input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>, output_gt : &microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize>,learning_rate : f32) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                self.predict_inner_train(input, output_gt, learning_rate).dequantize()
            }

            fn predict_inner_train(&mut self, input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>,output_gt : &microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize>, learning_rate: f32) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers_train
                #loss_ident
                #backward
                #output_ident
            }
            fn update_layers(&mut self, batch_size: usize, learning_rate: f32){
                #update
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}
