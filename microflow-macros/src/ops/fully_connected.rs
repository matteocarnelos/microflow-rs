use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

/// Represents the tokenized version of the `FullyConnected` operator.
pub(crate) struct TokenFullyConnected<T: TokenQuantized> {
    pub(crate) weights: TokenTensor2D<T>,
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>, TokenBuffer2D<i32>),
    pub(crate) index: usize,
}

/// Parses the [`TokenFullyConnected`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenFullyConnected::<i8>::new(
            operator, tensors, buffers, index,
        )),
        TensorType::UINT8 => Box::new(TokenFullyConnected::<u8>::new(
            operator, tensors, buffers, index,
        )),
        input_type => abort_call_site!(
            "FullyConnected supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenFullyConnected<T> {
    /// Builds the [`TokenFullyConnected`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    /// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    /// * `index` - The operator index
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor2D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let weights =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        // (F6): the converter drops an all-zero bias and sets the optional
        // input to -1 (or omits it) — a zero constant restores it.
        let biases = if inputs.len() >= 3 && inputs.get(2) >= 0 {
            TokenTensor2D::<i32>::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers)
        } else {
            Self::zero_bias(weights.shape[1])
        };
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator
            .builtin_options_as_fully_connected_options()
            .unwrap();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        Self {
            weights,
            output,
            fused_activation: options.fused_activation_function().into(),
            constants,
            index,
        }
    }

    fn zero_bias(weights_cols: usize) -> TokenTensor2D<i32> {
        TokenTensor2D {
            buffer: TokenBuffer2D::from(DMatrix::zeros(weights_cols, 1)),
            shape: vec![weights_cols, 1],
            scale: vec![1.0],
            zero_point: vec![0],
        }
    }

    /// Pre-processes the operator, returning the tuple of per-channel
    /// constants (one entry per output unit, spec §3.3):
    /// - `.0[j]`: the bias term, scaled to the output scale;
    /// - `.1[j]`: the multiplier `scale_x * scale_w[j] / scale_out`;
    /// - `.2[j]`: the zero-point correction
    ///   `zp_x * (sum_i w[i][j] - INPUT_COLS * zp_w[j])`.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `weights` - The weights of the operator as a [`TokenTensor2D`]
    /// * `biases` - The biases of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(
        input: &TokenTensor2D<T>,
        weights: &TokenTensor2D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor2D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>, TokenBuffer2D<i32>) {
        let weights_cols = weights.shape[1];
        (
            TokenBuffer2D::from(DMatrix::from_fn(weights_cols, 1, |j, _| {
                let bias_scale = biases.scale.get(j).copied().unwrap_or(biases.scale[0]);
                let bias_zero_point = biases
                    .zero_point
                    .get(j)
                    .copied()
                    .unwrap_or(biases.zero_point[0]);
                bias_scale / output.scale[0] * (biases.buffer[(j, 0)] - bias_zero_point) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(weights_cols, 1, |j, _| {
                input.scale[0] * weights.scale.get(j).copied().unwrap_or(weights.scale[0])
                    / output.scale[0]
            })),
            TokenBuffer2D::from(DMatrix::from_fn(1, weights_cols, |_, j| {
                let weights_zero_point = i32::from_subset(
                    &weights
                        .zero_point
                        .get(j)
                        .copied()
                        .unwrap_or(weights.zero_point[0]),
                );
                let column_sum: i32 = convert_ref::<DMatrix<T>, DMatrix<i32>>(&weights.buffer)
                    .column(j)
                    .sum();
                i32::from_subset(&input.zero_point[0])
                    * (column_sum - weights.shape[0] as i32 * weights_zero_point)
            })),
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenFullyConnected<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights_ident = format_ident!("weights_{}", self.index);
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        let output_shape = &self.output.shape;
        let output_scale = self.output.scale[0];
        let output_zero_point = self.output.zero_point[0];
        let fused_activation = self.fused_activation;
        let (constants_0, constants_1, constants_2) = &self.constants;

        let ts = quote! {
            const #weights_ident: #weights_type = #weights;
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                microflow::ops::fully_connected(
                    input,
                    &#weights_ident,
                    [#output_scale],
                    [#output_zero_point],
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation,
                    },
                    (#constants_0, #constants_1, #constants_2)
            );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn setup() -> TokenFullyConnected<i8> {
        TokenFullyConnected {
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    1, 2, 3;
                    4, 5, 6
                ]),
                shape: vec![2, 3],
                scale: vec![0.7],
                zero_point: vec![8],
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![1, 3],
                scale: vec![0.9],
                zero_point: vec![10],
            },
            fused_activation: TokenFusedActivation::Relu,
            constants: (
                TokenBuffer2D::from(dmatrix![11., 12., 13.]),
                TokenBuffer2D::from(dmatrix![14., 15., 16.]),
                TokenBuffer2D::from(dmatrix![17, 18, 19]),
            ),
            index: 0,
        }
    }

    #[test]
    fn fully_connected_preprocess() {
        let layer = setup();
        let input = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: vec![0.17],
            zero_point: vec![18],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                19;
                20;
                21
            ]),
            shape: vec![3, 1],
            scale: vec![0.22],
            zero_point: vec![23],
        };
        let constants =
            TokenFullyConnected::preprocess(&input, &layer.weights, &biases, &layer.output);
        assert_eq!(
            constants.0 .0,
            Some(dmatrix![-0.9777778; -0.73333335; -0.4888889])
        );
        assert_eq!(
            constants.1 .0,
            Some(dmatrix![0.13222224; 0.13222224; 0.13222224])
        );
        assert_eq!(
            constants.2 .0,
            Some(dmatrix![90 - 288, 126 - 288, 162 - 288])
        );
    }

    #[test]
    fn fully_connected_preprocess_per_channel_weights() {
        let layer = setup();
        // Per-channel weights with the per-tensor values repeated: identical
        // constants (the per-channel path must degenerate to the per-tensor one).
        let weights = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                1, 2, 3;
                4, 5, 6
            ]),
            shape: vec![2, 3],
            scale: vec![0.7, 0.7, 0.7],
            zero_point: vec![8, 8, 8],
        };
        let input = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: vec![0.17],
            zero_point: vec![18],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                19;
                20;
                21
            ]),
            shape: vec![3, 1],
            scale: vec![0.22],
            zero_point: vec![23],
        };
        let constants = TokenFullyConnected::preprocess(&input, &weights, &biases, &layer.output);
        assert_eq!(
            constants.1 .0,
            Some(dmatrix![0.13222224; 0.13222224; 0.13222224])
        );
        assert_eq!(
            constants.2 .0,
            Some(dmatrix![90 - 288, 126 - 288, 162 - 288])
        );
    }

    #[test]
    fn fully_connected_to_tokens() {
        let layer = setup();
        let weights = &layer.weights;
        let fused_activation = layer.fused_activation;
        let constants_0 = &layer.constants.0;
        let constants_1 = &layer.constants.1;
        let constants_2 = &layer.constants.2;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const weights_0: microflow::tensor::Tensor2D<i8, 2usize, 3usize, 1usize> = #weights;
                let input: microflow::tensor::Tensor2D<_, 1usize, 3usize, 1usize> =
                    microflow::ops::fully_connected(
                        input,
                        &weights_0,
                        [0.9f32],
                        [10i8],
                        microflow::ops::FullyConnectedOptions {
                            fused_activation: #fused_activation,
                        },
                        (#constants_0, #constants_1, #constants_2)
                );
            }
            .to_string()
        );
    }
}
