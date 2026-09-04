//! [![crates.io](https://img.shields.io/crates/v/microflow-macros)](https://crates.io/crates/microflow-macros)
//! [![docs.rs](https://img.shields.io/docsrs/microflow-macros)](https://docs.rs/microflow-macros)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! Macro crate of the [MicroFlow](https://github.com/matteocarnelos/microflow-rs) inference engine, namely, the MicroFlow compiler.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, ItemStruct};

use crate::shape_fold::{conversion_tokens, fold};
use crate::tflite_flatbuffers::tflite::{BuiltinOperator, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use ops::*;
use structmeta::StructMeta;
use syn::LitStr;
use tflite_flatbuffers::tflite::root_as_model;

mod activation;
mod buffer;
mod ops;
mod quantize;
mod shape_fold;
mod tensor;
#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
// flatc-generated code: the upstream generator predates the 2024 lifetime
// elision rules, so `Model` is written without `<'_>`. The local CI passes
// `-A mismatched_lifetime_syntaxes` for clippy; this keeps plain local
// `cargo build` runs (scripts/qemu-parity.sh etc.) equally quiet.
#[allow(mismatched_lifetime_syntaxes)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
}

/// Is this CONV_2D the 1-D case (OHWI filters with height 1 and no vertical
/// stride, §2.3)? The input-height half of the check lives in the caller,
/// which sees the effective (folded) input shape.
fn is_conv_1d(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> bool {
    let inputs = operator.inputs().unwrap_or_default();
    if inputs.len() < 2 || inputs.get(1) < 0 {
        return false;
    }
    let filters_height_1 = tensors
        .get(inputs.get(1) as usize)
        .shape()
        .map(|shape| shape.len() == 4 && shape.get(1) == 1)
        .unwrap_or(false);
    let stride_h_1 = operator
        .builtin_options_as_conv_2_doptions()
        .map(|options| options.stride_h() == 1)
        .unwrap_or(true);
    filters_height_1 && stride_h_1
}

/// Week-6 benchmark escape hatch: `MICROFLOW_CONV2D_ONLY=1` forces every
/// 1-D convolution onto the generic `conv_2d` kernel (the pre-Conv1D
/// "reshape trick" path) for a same-model A/B of the dedicated kernel —
/// footprint (scripts/footprint.sh) and criterion (benches/conv1d.rs).
///
/// Build trap: the variable is read at macro-expansion time and cargo does
/// NOT fingerprint proc-macro env reads — an A/B build must use its own
/// `CARGO_TARGET_DIR` (stale artifacts otherwise silently keep the old
/// path; the week-6 twin of the week-3 toolchain trap in ../NOTES.md).
fn force_conv_2d() -> bool {
    std::env::var("MICROFLOW_CONV2D_ONLY").as_deref() == Ok("1")
}

/// The entry point of MicroFlow.
/// This attribute-like procedural macro can be placed on `structs` to implement the `predict()`
/// function based on the given model.
/// The macro takes as input the path of the model, which must be in the TensorFlow Lite format
/// (`.tflite`).
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

    let ident = &item.ident;

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();
    let input_tensor_index = subgraph.inputs().unwrap().get(0) as usize;
    let output_tensor_index = subgraph.outputs().unwrap().get(0) as usize;

    let input = tensors.get(input_tensor_index);
    let raw_input_shape: Vec<usize> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    let input_type = match input.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        input_type => abort_call_site!(
            "unsupported input tensor type: {:?}. Supported input types are INT8 and UINT8",
            input_type
        ),
    };
    // §2.2: a rank-3 (1, T, C) input is normalized to (1, 1, T, C); the
    // user-facing type stays 2-D (T, C) — the batch axis does not enter the
    // user type.
    let (input_shape, user_input_shape, rank3_input) = match raw_input_shape.len() {
        1 => {
            let mut shape = raw_input_shape.clone();
            shape.insert(0, 1);
            (shape.clone(), shape, false)
        }
        2 => (raw_input_shape.clone(), raw_input_shape.clone(), false),
        3 if raw_input_shape[0] == 1 => {
            let mut shape = raw_input_shape.clone();
            shape.insert(1, 1);
            (shape, vec![raw_input_shape[1], raw_input_shape[2]], true)
        }
        3 => abort_call_site!(
            "unsupported input tensor rank: 3 with shape {:?}: the batch dimension (the first \
             one) must be 1 for rank-3 inputs",
            raw_input_shape
        ),
        4 => (raw_input_shape.clone(), raw_input_shape.clone(), false),
        rank => abort_call_site!(
            "unsupported input tensor rank: {} (shape {:?}). Supported ranks are 2, 3 (batch 1) and 4",
            rank,
            raw_input_shape
        ),
    };
    let input_tensor = match input_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        rank => abort_call_site!("unsupported input tensor rank: {}", rank),
    };
    let input_buffer = if input_shape.len() == 4 && !rank3_input {
        quote!(Buffer4D)
    } else {
        quote!(Buffer2D)
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
        input_type => abort_call_site!(
            "unsupported input zero-point tensor type: {:?}. Supported types are INT8 and UINT8",
            input_type
        ),
    };

    // §2.1: fold the shape operators (EXPAND_DIMS/RESHAPE/SHAPE chains) away
    // and collect the real operators with their effective shapes.
    let operators = subgraph.operators().unwrap();
    let operator_codes = model.operator_codes().unwrap();
    let folded = fold(
        operators,
        operator_codes,
        tensors,
        buffers,
        input_tensor_index,
        output_tensor_index,
        &raw_input_shape,
    )
    .unwrap_or_else(|error| abort_call_site!("model normalization failed: {}", error));

    let mut layers = TokenStream2::new();
    let mut current_shape = input_shape.clone();
    for op in &folded.ops {
        // A folded RESHAPE between two real operators becomes a conversion.
        if current_shape != op.input_shape {
            let conversion =
                conversion_tokens(&current_shape, &op.input_shape).unwrap_or_else(|error| {
                    abort_call_site!("{:?} (op {}): {error}", op.kind, op.index)
                });
            conversion.to_tokens(&mut layers);
        }
        let operator = operators.get(op.index);
        let layer: Box<dyn ToTokens> = match op.kind {
            BuiltinOperator::FULLY_CONNECTED => {
                fully_connected::parse(operator, tensors, buffers, op.index)
            }
            BuiltinOperator::DEPTHWISE_CONV_2D => {
                depthwise_conv_2d::parse(operator, tensors, buffers, op.index, &op.output_shape)
            }
            BuiltinOperator::CONV_2D => {
                // §2.3: the 1-D case is CONV_2D over a (1, 1, T, C) input with
                // height-1 filters — it goes through the dedicated week-2
                // conv_1d kernel; everything else stays on the conv_2d path.
                let one_d_input = op.input_shape.len() == 4 && op.input_shape[1] == 1;
                if one_d_input && is_conv_1d(operator, tensors) && !force_conv_2d() {
                    conv_1d::parse(
                        operator,
                        tensors,
                        buffers,
                        op.index,
                        &op.input_shape,
                        &op.output_shape,
                    )
                } else {
                    conv_2d::parse(operator, tensors, buffers, op.index, &op.output_shape)
                }
            }
            BuiltinOperator::AVERAGE_POOL_2D => {
                average_pool_2d::parse(operator, tensors, &op.output_shape)
            }
            BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors, &op.output_shape),
            BuiltinOperator::TRANSPOSE => transpose::parse(operator, tensors, buffers),
            unsupported => abort_call_site!("unsupported operator: {:?}", unsupported),
        };
        layer.to_tokens(&mut layers);
        current_shape = op.output_shape.clone();
    }
    // A folded tail between the last real operator and the model output.
    if current_shape != folded.output_shape {
        let conversion =
            conversion_tokens(&current_shape, &folded.output_shape).unwrap_or_else(|error| {
                abort_call_site!("the model output tensor cannot be produced: {}", error)
            });
        conversion.to_tokens(&mut layers);
    }

    let output = tensors.get(output_tensor_index);
    let output_type = match output.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        output_type => abort_call_site!(
            "unsupported output tensor type: {:?}. Supported output types are INT8 and UINT8",
            output_type
        ),
    };
    let output_shape = folded.output_shape;
    let output_tensor = match output_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        rank => abort_call_site!("unsupported output tensor rank: {}", rank),
    };
    let output_buffer = if output_shape.len() == 4 {
        quote!(Buffer4D)
    } else {
        quote!(Buffer2D)
    };

    // §2.2: for a rank-3 input the user hands over a (T, C) 2-D buffer; it is
    // re-labeled into the internal (1, 1, T, C) representation without
    // touching the data order (a stack copy, no heap).
    let (timesteps, chans) = if rank3_input {
        (input_shape[2], input_shape[3])
    } else {
        (0, 0)
    };
    let predict_body_input = if rank3_input {
        quote! {
            let input = microflow::tensor::Tensor4D::<#input_type, 1usize, 1usize, #timesteps, #chans, 1usize>::quantize(
                [microflow::buffer::Buffer2D::<[f32; #chans], 1usize, #timesteps>::from_fn(|_, ts| {
                    core::array::from_fn(|c| input[(ts, c)])
                })],
                [#(#input_scale),*],
                [#(#input_zero_point),*],
            );
        }
    } else {
        quote! {
            let input = microflow::tensor::#input_tensor::quantize(
                input,
                [#(#input_scale),*],
                [#(#input_zero_point),*],
            );
        }
    };
    let predict_quantized_body_input = if rank3_input {
        quote! {
            let input = microflow::tensor::Tensor4D::<#input_type, 1usize, 1usize, #timesteps, #chans, 1usize>::new(
                [microflow::buffer::Buffer2D::<[#input_type; #chans], 1usize, #timesteps>::from_fn(|_, ts| {
                    core::array::from_fn(|c| input[(ts, c)])
                })],
                [#(#input_scale),*],
                [#(#input_zero_point),*],
            );
        }
    } else {
        quote! {
            let input = microflow::tensor::#input_tensor::new(
                input,
                [#(#input_scale),*],
                [#(#input_zero_point),*],
            );
        }
    };

    let ts = quote! {
        #item
        impl #ident {
            pub fn predict(input: microflow::buffer::#input_buffer<f32, #(#user_input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                #predict_body_input
                Self::predict_inner(input).dequantize()
            }

            pub fn predict_quantized(input: microflow::buffer::#input_buffer<#input_type, #(#user_input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                #predict_quantized_body_input
                Self::predict_inner(input).dequantize()
            }

            fn predict_inner(input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers
                input
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}
