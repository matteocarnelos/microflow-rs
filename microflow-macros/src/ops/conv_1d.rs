use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote, ToTokens};

/// Represents the tokenized version of a `Conv2D` operator restricted to the
/// 1-D case (filters height 1, spec §2.3): the week-2 `conv_1d` kernel is
/// called instead of the generic `conv_2d`.
pub(crate) struct TokenConv1D<T: TokenQuantized> {
    pub(crate) filters: TokenTensor4D<T>,
    /// Bias in accumulator units (§3.1): the kernel adds it to the i32
    /// accumulator directly.
    pub(crate) biases: TokenTensor2D<i32>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) stride: usize,
    pub(crate) index: usize,
}

/// Parses the [`TokenConv1D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
/// * `input_shape` - The effective input shape after folding, `(1, 1, T, C)`
/// * `output_shape` - The effective output shape after folding, `(1, 1, T', F)`
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenConv1D::<i8>::new(
            operator,
            tensors,
            buffers,
            index,
            input_shape,
            output_shape,
        )),
        TensorType::UINT8 => Box::new(TokenConv1D::<u8>::new(
            operator,
            tensors,
            buffers,
            index,
            input_shape,
            output_shape,
        )),
        input_type => abort_call_site!(
            "Conv1D supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

/// Number of output timesteps for the given geometry (mirrors the kernel).
const fn output_len(timesteps: usize, kernel: usize, stride: usize, same: bool) -> Option<usize> {
    if same {
        Some(timesteps.div_ceil(stride))
    } else if timesteps >= kernel {
        Some((timesteps - kernel) / stride + 1)
    } else {
        None
    }
}

impl<T: TokenQuantized> TokenConv1D<T> {
    /// Builds the [`TokenConv1D`] operator from the given model operator and
    /// tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    /// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    /// * `index` - The operator index
    /// * `input_shape` - The effective input shape after folding, `(1, 1, T, C)`
    /// * `output_shape` - The effective output shape after folding, `(1, 1, T', F)`
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor_with_shape(
            tensors.get(inputs.get(0) as usize),
            input_shape.to_vec(),
        );
        let filters =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor_with_shape(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
            output_shape.to_vec(),
        );
        let options = operator.builtin_options_as_conv_2_doptions().unwrap();

        // The 1-D contract (§2.3): OHWI filters with H == 1.
        let [filters_out, filters_height, kernel, filters_chans] = filters.shape[..] else {
            abort_call_site!(
                "Conv1D (op {}): expected rank-4 OHWI filters, got shape {:?}",
                index,
                filters.shape
            );
        };
        if filters_height != 1 {
            abort_call_site!(
                "Conv1D (op {}): filters height is {}, expected 1 (the 1-D serialization, \
                 fact F5); a real 2-D convolution must go through the conv_2d path",
                index,
                filters_height
            );
        }
        if input_shape.len() != 4 || input_shape[1] != 1 {
            abort_call_site!(
                "Conv1D (op {}): the input must be (1, 1, T, C), got {:?}",
                index,
                input_shape
            );
        }
        let timesteps = input_shape[2];
        let chans = input_shape[3];
        if filters_chans != chans {
            abort_call_site!(
                "Conv1D (op {}): filters expect {} channels, the input has {}",
                index,
                filters_chans,
                chans
            );
        }
        let stride = options.stride_w() as usize;
        if options.stride_h() as usize != 1 {
            abort_call_site!(
                "Conv1D (op {}): stride_h is {} (the 1-D case strides only over time)",
                index,
                options.stride_h()
            );
        }
        // Geometry check (§4.5): the folded output shape must match the
        // padding/stride arithmetic; an empty Valid output is rejected here.
        let same = options.padding() == crate::tflite_flatbuffers::tflite::Padding::SAME;
        let Some(expected_len) = output_len(timesteps, kernel, stride, same) else {
            abort_call_site!(
                "Conv1D (op {}): T ({}) < kernel ({}) with Valid padding would produce an \
                 empty output, which is forbidden",
                index,
                timesteps,
                kernel
            );
        };
        if output_shape.len() != 4 || output_shape[1] != 1 {
            abort_call_site!(
                "Conv1D (op {}): the output must be (1, 1, T', F), got {:?}",
                index,
                output_shape
            );
        }
        if output_shape[2] != expected_len || output_shape[3] != filters_out {
            abort_call_site!(
                "Conv1D (op {}): output shape {:?} does not match the geometry (timesteps {}, \
                 kernel {}, stride {}, padding {}): expected (1, 1, {}, {})",
                index,
                output_shape,
                timesteps,
                kernel,
                stride,
                if same { "same" } else { "valid" },
                expected_len,
                filters_out
            );
        }

        // (F6): the converter drops an all-zero bias — an explicit zero
        // constant restores it.
        let biases = if inputs.len() >= 3 && inputs.get(2) >= 0 {
            TokenTensor2D::<i32>::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers)
        } else {
            Self::zero_bias(filters_out)
        };
        let biases = Self::to_accumulator_units(&biases, &input, &filters);

        Self {
            filters,
            biases,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            stride,
            index,
        }
    }

    fn zero_bias(filters_out: usize) -> TokenTensor2D<i32> {
        TokenTensor2D {
            buffer: TokenBuffer2D::from(DMatrix::zeros(filters_out, 1)),
            shape: vec![filters_out, 1],
            scale: vec![1.0],
            zero_point: vec![0],
        }
    }

    /// Converts the INT32 bias from the file's units to accumulator units
    /// (§3.1). TFLite writes `scale_b[f] = scale_x * scale_w[f]`, so the ratio
    /// is ~1 and the round-trip returns the original value; anything else is
    /// rescaled instead of silently corrupting the result.
    fn to_accumulator_units(
        biases: &TokenTensor2D<i32>,
        input: &TokenTensor4D<T>,
        filters: &TokenTensor4D<T>,
    ) -> TokenTensor2D<i32> {
        let filters_out = filters.shape[0];
        // The kernel reads only the buffer values; the quant fields are unused
        // but must have one entry per filter for the tensor type to line up.
        let quants = filters.scale.len().max(1);
        let buffer = TokenBuffer2D::from(DMatrix::from_fn(filters_out, 1, |f, _| {
            let scale_w = filters.scale.get(f).copied().unwrap_or(filters.scale[0]) as f64;
            let scale_b = biases.scale.get(f).copied().unwrap_or(biases.scale[0]) as f64;
            let raw = biases.buffer[(f, 0)] as f64;
            (raw * scale_b / (input.scale[0] as f64 * scale_w)).round() as i32
        }));
        TokenTensor2D {
            buffer,
            shape: vec![filters_out, 1],
            scale: vec![1.0; quants],
            zero_point: vec![0; quants],
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenConv1D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let filters_ident = format_ident!("filters_{}", self.index);
        let biases_ident = format_ident!("biases_{}", self.index);
        let filters_type = self.filters.type_tokens();
        let filters_out = self.filters.shape[0];
        let quants = self.filters.scale.len().max(1);
        let filters = &self.filters;
        let biases = &self.biases;
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let stride = self.stride;

        let ts = quote! {
            const #filters_ident: #filters_type = #filters;
            const #biases_ident: microflow::tensor::Tensor2D<i32, #filters_out, 1usize, #quants> = #biases;
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::conv_1d(
                    input,
                    &#filters_ident,
                    &#biases_ident,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::Conv1DOptions {
                        fused_activation: #fused_activation,
                        padding: #view_padding,
                        stride: #stride,
                    },
            );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};

    fn setup() -> TokenConv1D<i8> {
        TokenConv1D {
            filters: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![vec![1], vec![2], vec![3]],
                    dmatrix![vec![4], vec![5], vec![6]],
                ]),
                shape: vec![2, 1, 3, 1],
                scale: vec![0.25, 0.26],
                zero_point: vec![0, 0],
            },
            biases: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![10; 20]),
                shape: vec![2, 1],
                scale: vec![1.0, 1.0],
                zero_point: vec![0, 0],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 1, 4, 2],
                scale: vec![0.29],
                zero_point: vec![-128],
            },
            fused_activation: TokenFusedActivation::Relu,
            view_padding: TokenTensorViewPadding::Valid,
            stride: 1,
            index: 1,
        }
    }

    #[test]
    fn conv_1d_to_tokens() {
        let layer = setup();
        let filters = &layer.filters;
        let biases = &layer.biases;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const filters_1: microflow::tensor::Tensor4D<i8, 2usize, 1usize, 3usize, 1usize, 2usize> = #filters;
                const biases_1: microflow::tensor::Tensor2D<i32, 2usize, 1usize, 2usize> = #biases;
                let input: microflow::tensor::Tensor4D<_, 1usize, 1usize, 4usize, 2usize, 1usize> =
                    microflow::ops::conv_1d(
                        input,
                        &filters_1,
                        &biases_1,
                        [0.29f32],
                        [-128i8],
                        microflow::ops::Conv1DOptions {
                            fused_activation: #fused_activation,
                            padding: #view_padding,
                            stride: 1usize,
                        },
                );
            }
            .to_string()
        )
    }

    #[test]
    fn conv_1d_geometry() {
        assert_eq!(output_len(128, 3, 1, false), Some(126));
        assert_eq!(output_len(126, 2, 2, false), Some(63));
        assert_eq!(output_len(2, 3, 1, false), None, "empty Valid output");
        assert_eq!(output_len(2, 3, 1, true), Some(2), "Same keeps T");
    }
}
