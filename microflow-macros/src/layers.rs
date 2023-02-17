use flatbuffers::{ForwardsUOffset, Vector};

use crate::tflite_flatbuffers::tflite::{ActivationFunctionType, Buffer, Operator, Tensor};

use super::*;

pub fn fully_connected(
    tensors: Vector<ForwardsUOffset<Tensor>>,
    operator: Operator,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> TokenStream2 {
    let weights = tensors.get(operator.inputs().unwrap().get(1) as usize);
    let weights_scale = weights.quantization().unwrap().scale().unwrap().get(0);
    let weights_zero_point = weights.quantization().unwrap().zero_point().unwrap().get(0) as i8;
    let weights_data: Vec<i8> = buffers
        .get(weights.buffer() as usize)
        .data()
        .unwrap()
        .iter()
        .map(|e| e as i8)
        .collect();
    let mut weights_rows: Vec<TokenStream2> = Vec::new();

    let weights_row_size = weights.shape().unwrap().get(0) as usize;
    let weights_column_size = weights.shape().unwrap().get(1) as usize;
    for i in 0..weights_column_size {
        let mut row: Vec<i8> = Vec::new();
        for j in 0..weights_row_size {
            row.push(weights_data[i + j * weights_column_size])
        }
        weights_rows.push(quote! {#(#row),*});
    }

    let biases = tensors.get(operator.inputs().unwrap().get(2) as usize);
    let biases_scale = biases.quantization().unwrap().scale().unwrap().get(0);
    let biases_zero_point = biases.quantization().unwrap().zero_point().unwrap().get(0) as i32;
    let biases_data: Vec<i32> = buffers
        .get(biases.buffer() as usize)
        .data()
        .unwrap()
        .bytes()
        .chunks(4)
        .map(|e| i32::from_be_bytes([e[3], e[2], e[1], e[0]]))
        .collect();

    let output = tensors.get(operator.outputs().unwrap().get(0) as usize);
    let output_scale = output.quantization().unwrap().scale().unwrap().get(0);
    let output_zero_point = output.quantization().unwrap().zero_point().unwrap().get(0) as i8;

    let activation = match operator
        .builtin_options_as_fully_connected_options()
        .unwrap()
        .fused_activation_function()
    {
        ActivationFunctionType::RELU => quote! { microflow::activations::Activation::RELU },
        ActivationFunctionType::NONE => quote! { microflow::activations::Activation::NONE },
        _ => unimplemented!(),
    };

    quote! {
        let weights = microflow::tensor::QuantizedTensor::new(
            nalgebra::matrix![#(#weights_rows);*],
            #weights_scale,
            #weights_zero_point
        );
        let biases = microflow::tensor::QuantizedTensor::new(
            nalgebra::vector![#(#biases_data),*],
            #biases_scale,
            #biases_zero_point
        );
        let output = microflow::ops::fully_connected(output, weights, biases, #output_scale, #output_zero_point, #activation);
    }
}
