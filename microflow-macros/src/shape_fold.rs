//! Graph normalization for rank-3 (Keras `Conv1D`) models — spec §2.1.
//!
//! A TFLite file exported from TF 2.x wraps every 1-D block into shape
//! operators: `EXPAND_DIMS → CONV_2D → RESHAPE`, pools the same way, and
//! flattens via a dynamic `SHAPE → STRIDED_SLICE → PACK → RESHAPE` chain
//! (facts F2–F4 of the spike). None of them touch the data: they only re-label
//! the shape. This pass walks the operator list once, keeps a
//! "tensor → virtual shape" table, and:
//!
//! - folds `EXPAND_DIMS`/`RESHAPE` into virtual reshapes (no code emitted);
//! - evaluates the flatten chain statically and replaces it with a single
//!   virtual reshape;
//! - keeps the "real" operators (`CONV_2D`, `AVERAGE_POOL_2D`,
//!   `FULLY_CONNECTED`, `SOFTMAX`, `DEPTHWISE_CONV_2D`, `TRANSPOSE`) with
//!   effective input/output shapes derived from the table.
//!
//! Anything that does not fold produces an `Err` with the operator, its
//! shapes, and what was expected; the macro aborts with that message (§2.1).

use std::mem::size_of;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

use crate::tflite_flatbuffers::tflite::{Buffer, BuiltinOperator, Operator, OperatorCode, Tensor};
use flatbuffers::{ForwardsUOffset, Vector};

/// A real (code-generating) operator with the shapes it sees after folding.
pub(crate) struct FoldedOp {
    /// Index in the TFLite operator list (used for naming generated constants).
    pub(crate) index: usize,
    pub(crate) kind: BuiltinOperator,
    /// Effective shape of input tensor 0, normalized to rank 2 or 4.
    pub(crate) input_shape: Vec<usize>,
    /// Effective shape of output tensor 0, normalized to rank 2 or 4.
    pub(crate) output_shape: Vec<usize>,
}

/// The result of folding a subgraph.
pub(crate) struct FoldedGraph {
    /// Real operators, in execution order.
    pub(crate) ops: Vec<FoldedOp>,
    /// Virtual shape of the subgraph output tensor, rank 2 or 4.
    pub(crate) output_shape: Vec<usize>,
}

/// What the folding pass knows about a tensor.
#[derive(Clone)]
enum Value {
    /// Data produced by real op output `origin` (or by the model input), with
    /// the virtual shape accumulated across the folded shape operators.
    Data { origin: usize, shape: Vec<usize> },
    /// A statically-known shape vector (a result of the `SHAPE` chain).
    Dims(Vec<i64>),
}

/// Resolves the builtin operator kind of an operator (codes < 128 live in the
/// deprecated byte field; newer ones only in `builtin_code`).
fn builtin_kind(
    codes: Vector<ForwardsUOffset<OperatorCode>>,
    operator: Operator,
) -> BuiltinOperator {
    let code = codes.get(operator.opcode_index() as usize);
    let builtin = code.builtin_code();
    if builtin == BuiltinOperator::ADD && code.deprecated_builtin_code() != 0 {
        BuiltinOperator(code.deprecated_builtin_code() as i32)
    } else {
        builtin
    }
}

fn tensor_name(tensor: &Tensor) -> String {
    tensor.name().unwrap_or("<unnamed>").to_string()
}

fn tensor_shape(tensor: &Tensor) -> Vec<usize> {
    tensor
        .shape()
        .unwrap_or_default()
        .iter()
        .map(|e| e as usize)
        .collect()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Reads the INT32 little-endian constant buffer of a tensor.
fn read_i32s(
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Result<Vec<i32>, String> {
    let tensor = tensors.get(index);
    let name = tensor_name(&tensor);
    let expected = tensor_shape(&tensor).iter().product::<usize>();
    let data = buffers
        .get(tensor.buffer() as usize)
        .data()
        .ok_or_else(|| format!("constant tensor '{name}' (#{index}) has no buffer"))?;
    let bytes = data.bytes();
    if bytes.len() % size_of::<i32>() != 0 || bytes.len() / size_of::<i32>() != expected {
        return Err(format!(
            "constant tensor '{name}' (#{index}) declares {expected} values but its buffer holds {}",
            bytes.len() / size_of::<i32>()
        ));
    }
    Ok(bytes
        .chunks_exact(size_of::<i32>())
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i32_scalar(
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Result<i32, String> {
    let values = read_i32s(tensors, buffers, index)?;
    values.first().copied().ok_or_else(|| {
        format!(
            "constant tensor '{}' (#{index}) is empty, a scalar was expected",
            tensor_name(&tensors.get(index))
        )
    })
}

/// The data input of an operator: returns its origin and virtual shape.
fn data_input(
    op: &str,
    index: usize,
    tensor_index: i32,
    values: &[Option<Value>],
) -> Result<(usize, Vec<usize>), String> {
    match values
        .get(tensor_index.max(0) as usize)
        .and_then(|v| v.as_ref())
    {
        Some(Value::Data { origin, shape }) => Ok((*origin, shape.clone())),
        Some(Value::Dims(_)) => Err(format!(
            "{op} (op {index}): the input is a shape tensor, but data was expected"
        )),
        None => Err(format!(
            "{op} (op {index}): input tensor #{tensor_index} is not produced inside the \
             folded chain (constants and branching graphs are not supported)"
        )),
    }
}

/// Resolves a target shape that may contain a single `-1` dimension.
pub(crate) fn resolve_shape(dims: &[i64], old_len: usize) -> Result<Vec<usize>, String> {
    let unknowns = dims.iter().filter(|&&d| d == -1).count();
    if unknowns > 1 {
        return Err("at most one -1 dimension is allowed".into());
    }
    let known: i64 = dims.iter().filter(|&&d| d > 0).product();
    let total = old_len as i64;
    let resolved: Vec<i64> = dims
        .iter()
        .map(|&d| {
            if d == -1 {
                if known == 0 || total % known != 0 {
                    Err(format!("{total} elements do not divide evenly by {known}"))
                } else {
                    Ok(total / known)
                }
            } else if d <= 0 {
                Err(format!(
                    "invalid dimension {d} (only positive sizes and one -1 are allowed)"
                ))
            } else {
                Ok(d)
            }
        })
        .collect::<Result<_, _>>()?;
    if resolved.iter().product::<i64>() != total {
        return Err(format!(
            "the new shape {resolved:?} holds {} elements, but the input has {total}",
            resolved.iter().product::<i64>()
        ));
    }
    Ok(resolved.iter().map(|&d| d as usize).collect())
}

/// Inserts a size-1 axis (EXPAND_DIMS semantics), with negative-axis support.
pub(crate) fn expand_shape(shape: &[usize], axis: i32) -> Result<Vec<usize>, String> {
    let rank = shape.len() as i32;
    let resolved = if axis < 0 { axis + rank + 1 } else { axis };
    if !(0..=rank).contains(&resolved) {
        return Err(format!(
            "axis {axis} is out of range for a rank-{rank} tensor"
        ));
    }
    let mut expanded = shape.to_vec();
    expanded.insert(resolved as usize, 1);
    Ok(expanded)
}

/// Evaluates a 1-D strided slice over a statically-known shape vector.
/// `shrink` selects a single element (a scalar result), like the flatten
/// chain's `STRIDED_SLICE(shape, [0], [1], [1])`.
pub(crate) fn strided_slice(
    dims: &[i64],
    begin: &[i32],
    end: &[i32],
    strides: &[i32],
    shrink: bool,
) -> Result<Vec<i64>, String> {
    if begin.len() != 1 || end.len() != 1 || strides.len() != 1 {
        return Err(
            "only 1-D slices of statically-known shape vectors are supported \
             (begin/end/strides must all have exactly one element)"
                .into(),
        );
    }
    let len = dims.len() as i32;
    let stride = strides[0];
    if stride == 0 {
        return Err("stride must not be zero".into());
    }
    let b = (if begin[0] < 0 {
        begin[0] + len
    } else {
        begin[0]
    })
    .clamp(0, len);
    let e = (if end[0] < 0 { end[0] + len } else { end[0] }).clamp(0, len);
    let mut selected = Vec::new();
    let mut i = b;
    while if stride > 0 { i < e } else { i > e } {
        if i >= 0 && i < len {
            selected.push(dims[i as usize]);
        }
        i += stride;
    }
    if shrink && selected.len() != 1 {
        return Err(format!(
            "a shrink-axis slice must select exactly one element, got {}",
            selected.len()
        ));
    }
    Ok(selected)
}

/// Normalizes the output shape of a real operator to rank 2 or 4.
fn normalize_real_output(kind: BuiltinOperator, raw: Vec<usize>) -> Result<Vec<usize>, String> {
    let conv_like = matches!(
        kind,
        BuiltinOperator::CONV_2D
            | BuiltinOperator::DEPTHWISE_CONV_2D
            | BuiltinOperator::AVERAGE_POOL_2D
    );
    match raw.len() {
        1 => Ok([1, raw[0]].to_vec()),
        2 | 4 => Ok(raw),
        3 if raw[0] == 1 && conv_like => Ok([1, 1, raw[1], raw[2]].to_vec()),
        3 if raw[0] == 1 => Ok([raw[1], raw[2]].to_vec()),
        rank => Err(format!(
            "unsupported output tensor rank {rank} with shape {raw:?}"
        )),
    }
}

/// Normalizes the input shape of a real operator to rank 2 or 4.
fn normalize_real_input(kind: BuiltinOperator, raw: Vec<usize>) -> Result<Vec<usize>, String> {
    let conv_like = matches!(
        kind,
        BuiltinOperator::CONV_2D
            | BuiltinOperator::DEPTHWISE_CONV_2D
            | BuiltinOperator::AVERAGE_POOL_2D
    );
    if conv_like {
        match raw.len() {
            4 => Ok(raw),
            3 if raw[0] == 1 => {
                let mut shape = raw;
                shape.insert(1, 1);
                Ok(shape)
            }
            rank => Err(format!(
                "the input must be a rank-4 (batch, 1, timesteps, channels) tensor, \
                 got rank {rank} with shape {raw:?}"
            )),
        }
    } else {
        match raw.len() {
            1 => Ok(vec![1, raw[0]]),
            2 => Ok(raw),
            // A reshape chain may leave a higher-rank tensor at the FC/softmax
            // boundary — the data is flattened row-major, as RESHAPE does.
            rank if rank > 2 => Ok(vec![1, product(&raw)]),
            rank => Err(format!(
                "unsupported input tensor rank {rank} with shape {raw:?}"
            )),
        }
    }
}

/// Normalizes the subgraph output shape (rank 2 or 4; a rank-3 `(1, a, b)`
/// output is presented to the user as the flattened `(1, a*b)`).
pub(crate) fn normalize_model_output(raw: &[usize]) -> Result<Vec<usize>, String> {
    match raw.len() {
        1 => Ok([1, raw[0]].to_vec()),
        2 | 4 => Ok(raw.to_vec()),
        3 if raw[0] == 1 => Ok([1, raw[1] * raw[2]].to_vec()),
        rank => Err(format!(
            "unsupported output tensor rank {rank} with shape {raw:?}; \
             supported ranks are 2, 3 (batch 1) and 4"
        )),
    }
}

/// Folds the shape operators of a subgraph away (§2.1).
///
/// # Arguments
/// * `operators` - The subgraph operators
/// * `operator_codes` - The model operator codes (kind resolution)
/// * `tensors` - The subgraph tensors
/// * `buffers` - The model buffers (constant shape vectors)
/// * `input_tensor_index` - Index of the subgraph input tensor
/// * `output_tensor_index` - Index of the subgraph output tensor
/// * `raw_input_shape` - The input shape as declared in the file (§2.2
///   normalization is applied by the caller, on the generated chain types)
///
pub(crate) fn fold(
    operators: Vector<ForwardsUOffset<Operator>>,
    operator_codes: Vector<ForwardsUOffset<OperatorCode>>,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    input_tensor_index: usize,
    output_tensor_index: usize,
    raw_input_shape: &[usize],
) -> Result<FoldedGraph, String> {
    let mut values: Vec<Option<Value>> = vec![None; tensors.len()];
    values[input_tensor_index] = Some(Value::Data {
        origin: input_tensor_index,
        shape: raw_input_shape.to_vec(),
    });
    // The tensor index whose data the generated chain variable currently holds.
    let mut current_origin = input_tensor_index;
    let mut ops = Vec::new();

    for (index, operator) in operators.iter().enumerate() {
        let kind = builtin_kind(operator_codes, operator);
        let inputs = operator.inputs().unwrap_or_default();
        let outputs = operator.outputs().unwrap_or_default();
        let output_index = outputs.get(0).max(0) as usize;

        match kind {
            BuiltinOperator::EXPAND_DIMS => {
                let (origin, shape) = data_input("EXPAND_DIMS", index, inputs.get(0), &values)?;
                if inputs.len() < 2 || inputs.get(1) < 0 {
                    return Err(format!(
                        "EXPAND_DIMS (op {index}): a scalar axis input tensor is required"
                    ));
                }
                let axis = read_i32_scalar(tensors, buffers, inputs.get(1) as usize)
                    .map_err(|e| format!("EXPAND_DIMS (op {index}): {e}"))?;
                let shape = expand_shape(&shape, axis)
                    .map_err(|e| format!("EXPAND_DIMS (op {index}): {e}, input shape {shape:?}"))?;
                values[output_index] = Some(Value::Data { origin, shape });
            }
            BuiltinOperator::RESHAPE => {
                let (origin, shape) = data_input("RESHAPE", index, inputs.get(0), &values)?;
                let dims: Vec<i64> = if inputs.len() > 1 && inputs.get(1) >= 0 {
                    match &values[inputs.get(1) as usize] {
                        // The shape comes from the statically evaluated
                        // SHAPE/STRIDED_SLICE/PACK chain (F4).
                        Some(Value::Dims(dims)) => dims.clone(),
                        // A constant shape vector in the tensor's buffer.
                        _ => read_i32s(tensors, buffers, inputs.get(1) as usize)?
                            .iter()
                            .map(|&v| v as i64)
                            .collect(),
                    }
                } else {
                    operator
                        .builtin_options_as_reshape_options()
                        .and_then(|o| o.new_shape())
                        .ok_or_else(|| {
                            format!("RESHAPE (op {index}): neither a shape input nor a new_shape option")
                        })?
                        .iter()
                        .map(|v| v as i64)
                        .collect()
                };
                let new_shape = resolve_shape(&dims, product(&shape)).map_err(|e| {
                    format!("RESHAPE (op {index}): {e}, input shape {shape:?}, target {dims:?}")
                })?;
                values[output_index] = Some(Value::Data { origin, shape: new_shape });
            }
            BuiltinOperator::SHAPE => {
                let (_, shape) = data_input("SHAPE", index, inputs.get(0), &values)?;
                values[output_index] = Some(Value::Dims(shape.iter().map(|&s| s as i64).collect()));
            }
            BuiltinOperator::STRIDED_SLICE => {
                if inputs.len() < 4 || (0..4).any(|i| inputs.get(i) < 0) {
                    return Err(format!(
                        "STRIDED_SLICE (op {index}): begin/end/strides input tensors are required"
                    ));
                }
                let dims = match values.get(inputs.get(0) as usize).and_then(|v| v.as_ref())
                {
                    Some(Value::Dims(dims)) => dims.clone(),
                    _ => {
                        return Err(format!(
                            "STRIDED_SLICE (op {index}): only shape tensors produced by SHAPE \
                             are supported (the Flatten chain, F4)"
                        ))
                    }
                };
                let begin = read_i32s(tensors, buffers, inputs.get(1) as usize)
                    .map_err(|e| format!("STRIDED_SLICE (op {index}): {e}"))?;
                let end = read_i32s(tensors, buffers, inputs.get(2) as usize)
                    .map_err(|e| format!("STRIDED_SLICE (op {index}): {e}"))?;
                let strides = read_i32s(tensors, buffers, inputs.get(3) as usize)
                    .map_err(|e| format!("STRIDED_SLICE (op {index}): {e}"))?;
                let options = operator
                    .builtin_options_as_strided_slice_options()
                    .ok_or_else(|| format!("STRIDED_SLICE (op {index}): missing options"))?;
                if options.begin_mask() != 0
                    || options.end_mask() != 0
                    || options.ellipsis_mask() != 0
                    || options.new_axis_mask() != 0
                {
                    return Err(format!(
                        "STRIDED_SLICE (op {index}): only shrink_axis_mask is supported \
                         (begin/end/ellipsis/new_axis masks must be 0)"
                    ));
                }
                let sliced = strided_slice(&dims, &begin, &end, &strides, options.shrink_axis_mask() != 0)
                    .map_err(|e| format!("STRIDED_SLICE (op {index}): {e}, shape vector {dims:?}"))?;
                values[output_index] = Some(Value::Dims(sliced));
            }
            BuiltinOperator::PACK => {
                let axis = operator
                    .builtin_options_as_pack_options()
                    .map(|o| o.axis())
                    .unwrap_or(0);
                if axis != 0 {
                    return Err(format!(
                        "PACK (op {index}): only axis 0 is supported, got {axis} (the Flatten chain, F4)"
                    ));
                }
                let mut packed = Vec::with_capacity(inputs.len());
                for i in 0..inputs.len() {
                    let tensor_index = inputs.get(i);
                    let value = match values.get(tensor_index.max(0) as usize)
                        .and_then(|v| v.as_ref())
                    {
                        Some(Value::Dims(dims)) if dims.len() == 1 => dims[0],
                        _ => read_i32_scalar(tensors, buffers, tensor_index as usize)? as i64,
                    };
                    packed.push(value);
                }
                values[output_index] = Some(Value::Dims(packed));
            }
            kind @ (BuiltinOperator::CONV_2D
            | BuiltinOperator::DEPTHWISE_CONV_2D
            | BuiltinOperator::AVERAGE_POOL_2D
            | BuiltinOperator::FULLY_CONNECTED
            | BuiltinOperator::SOFTMAX
            | BuiltinOperator::TRANSPOSE) => {
                let (origin, raw_input) = data_input("real operator", index, inputs.get(0), &values)?;
                if origin != current_origin {
                    return Err(format!(
                        "{kind:?} (op {index}): the input tensor does not follow the previous \
                         operator's output — branching graphs are not supported"
                    ));
                }
                let input_shape = normalize_real_input(kind, raw_input)
                    .map_err(|e| format!("{kind:?} (op {index}): {e}"))?;
                let output_shape = normalize_real_output(
                    kind,
                    tensor_shape(&tensors.get(output_index)),
                )
                .map_err(|e| format!("{kind:?} (op {index}): {e}"))?;
                ops.push(FoldedOp {
                    index,
                    kind,
                    input_shape: input_shape.clone(),
                    output_shape: output_shape.clone(),
                });
                values[output_index] = Some(Value::Data {
                    origin: output_index,
                    shape: output_shape,
                });
                current_origin = output_index;
            }
            unsupported => {
                return Err(format!(
                    "unsupported operator {unsupported:?} (op {index}); supported: CONV_2D, \
                     DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, SOFTMAX, TRANSPOSE and \
                     the foldable shape operators (EXPAND_DIMS, RESHAPE, SHAPE, STRIDED_SLICE, PACK)"
                ))
            }
        }
    }

    let output_shape = match values.get(output_tensor_index).and_then(|v| v.as_ref()) {
        Some(Value::Data { shape, .. }) => normalize_model_output(shape)?,
        _ => return Err("the subgraph output tensor is not produced by the folded chain".into()),
    };

    Ok(FoldedGraph { ops, output_shape })
}

/// Emits one `let input: TensorN<_, shape..., 1> = reshape(input);` binding.
fn emit_reshape(tokens: &mut TokenStream2, shape: &[usize]) {
    let tensor = if shape.len() == 2 {
        quote!(Tensor2D)
    } else {
        quote!(Tensor4D)
    };
    let ts = quote! {
        let input: microflow::tensor::#tensor<_, #(#shape),*, 1usize> =
            microflow::ops::reshape(input);
    };
    ts.to_tokens(tokens);
}

/// Emits the conversions needed between two layers whose effective shapes
/// differ (a folded RESHAPE). Rank changes go through the runtime `reshape`
/// (the `From` impls between `Tensor2D` and `Tensor4D`); a 4D→4D reshape hops
/// through the flattened 2D form.
pub(crate) fn conversion_tokens(from: &[usize], to: &[usize]) -> Result<TokenStream2, String> {
    if from == to {
        return Ok(TokenStream2::new());
    }
    let n = product(from);
    if n != product(to) {
        return Err(format!(
            "cannot reshape {from:?} into {to:?}: {} elements vs {}",
            n,
            product(to)
        ));
    }
    let mut tokens = TokenStream2::new();
    let mut current = from.to_vec();
    if current.len() == 4 && (to.len() == 2 || current != to) {
        if to.len() == 2 && to[0] != current[0] {
            return Err(format!(
                "cannot reshape {from:?} into {to:?}: the batch dimension must be preserved"
            ));
        }
        let flat = vec![current[0], n / current[0]];
        emit_reshape(&mut tokens, &flat);
        current = flat;
    }
    if to.len() == 4 && current.len() == 2 {
        if current[0] != to[0] {
            return Err(format!(
                "cannot reshape {from:?} into {to:?}: the batch dimension must be preserved"
            ));
        }
        emit_reshape(&mut tokens, to);
        current = to.to_vec();
    }
    if current != *to {
        return Err(format!(
            "cannot reshape {from:?} into {to:?}: this rank/shape combination is not supported \
             (2D targets must be batch-major `(1, n)`)"
        ));
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tflite_flatbuffers::tflite::root_as_model;

    #[test]
    fn resolves_minus_one() {
        assert_eq!(resolve_shape(&[-1, 126, 8], 1008).unwrap(), vec![1, 126, 8]);
        assert_eq!(resolve_shape(&[1, 480], 480).unwrap(), vec![1, 480]);
        assert!(resolve_shape(&[-1, 5], 12).is_err(), "12 % 5 != 0");
        assert!(resolve_shape(&[-1, -1, 8], 64).is_err(), "two -1s");
        assert!(resolve_shape(&[3, 3], 8).is_err(), "element count mismatch");
    }

    #[test]
    fn expands_dims() {
        assert_eq!(expand_shape(&[1, 128, 1], -3).unwrap(), vec![1, 1, 128, 1]);
        assert_eq!(expand_shape(&[1, 126, 8], 1).unwrap(), vec![1, 1, 126, 8]);
        assert_eq!(expand_shape(&[3], 1).unwrap(), vec![3, 1]);
        assert!(expand_shape(&[1, 2], 3).is_err(), "axis out of range");
        assert!(
            expand_shape(&[1, 2], -4).is_err(),
            "negative axis out of range"
        );
    }

    #[test]
    fn slices_like_the_flatten_chain() {
        // The Flatten chain of the spike: shape[0:1:1] of [1, 30, 16] -> [1]
        assert_eq!(
            strided_slice(&[1, 30, 16], &[0], &[1], &[1], true).unwrap(),
            vec![1]
        );
        assert_eq!(
            strided_slice(&[1, 2, 3], &[1], &[3], &[1], false).unwrap(),
            vec![2, 3]
        );
        assert_eq!(
            strided_slice(&[1, 2, 3], &[3], &[0], &[-1], false).unwrap(),
            vec![3, 2]
        );
        assert_eq!(
            strided_slice(&[1, 2], &[-2], &[-1], &[1], false).unwrap(),
            vec![1]
        );
        assert!(strided_slice(&[1, 2], &[0], &[2], &[1], true).is_err());
    }

    #[test]
    fn normalizes_io_shapes() {
        assert_eq!(
            normalize_real_input(BuiltinOperator::CONV_2D, vec![1, 8, 4]).unwrap(),
            vec![1, 1, 8, 4]
        );
        assert!(normalize_real_input(BuiltinOperator::CONV_2D, vec![8, 4]).is_err());
        assert_eq!(
            normalize_real_input(BuiltinOperator::FULLY_CONNECTED, vec![1, 1, 30, 16]).unwrap(),
            vec![1, 480]
        );
        assert_eq!(
            normalize_real_output(BuiltinOperator::CONV_2D, vec![1, 30, 16]).unwrap(),
            vec![1, 1, 30, 16]
        );
        assert_eq!(
            normalize_real_output(BuiltinOperator::SOFTMAX, vec![1, 4]).unwrap(),
            vec![1, 4]
        );
        assert_eq!(normalize_model_output(&[1, 30, 16]).unwrap(), vec![1, 480]);
    }

    /// The week-1 spike model: 18 operators fold into exactly 6 layers (DoD §6).
    #[test]
    fn folds_spike_graph_to_six_layers() {
        let buf = std::fs::read("../models/conv1d.tflite")
            .expect("run from the microflow-macros crate directory");
        let model = root_as_model(&buf).unwrap();
        let subgraph = model.subgraphs().unwrap().get(0);
        let tensors = subgraph.tensors().unwrap();
        let input_index = subgraph.inputs().unwrap().get(0) as usize;
        let output_index = subgraph.outputs().unwrap().get(0) as usize;

        let folded = fold(
            subgraph.operators().unwrap(),
            model.operator_codes().unwrap(),
            tensors,
            model.buffers().unwrap(),
            input_index,
            output_index,
            // The raw (1, T, C) input shape: this pass tracks the file's
            // shapes; the §2.2 input normalization is applied by the caller.
            &[1, 128, 1],
        )
        .unwrap();

        let kinds: Vec<i32> = folded.ops.iter().map(|o| o.kind.0).collect();
        assert_eq!(
            kinds,
            vec![
                BuiltinOperator::CONV_2D.0,
                BuiltinOperator::AVERAGE_POOL_2D.0,
                BuiltinOperator::CONV_2D.0,
                BuiltinOperator::AVERAGE_POOL_2D.0,
                BuiltinOperator::FULLY_CONNECTED.0,
                BuiltinOperator::SOFTMAX.0,
            ],
            "18 ops -> 6 layers"
        );
        let shapes: Vec<(&[usize], &[usize])> = folded
            .ops
            .iter()
            .map(|o| (o.input_shape.as_slice(), o.output_shape.as_slice()))
            .collect();
        assert_eq!(shapes[0], (&[1, 1, 128, 1][..], &[1, 1, 126, 8][..]));
        assert_eq!(shapes[1], (&[1, 1, 126, 8][..], &[1, 1, 63, 8][..]));
        assert_eq!(shapes[2], (&[1, 1, 63, 8][..], &[1, 1, 61, 16][..]));
        assert_eq!(shapes[3], (&[1, 1, 61, 16][..], &[1, 1, 30, 16][..]));
        assert_eq!(
            shapes[4],
            (&[1, 480][..], &[1, 4][..]),
            "the folded Flatten"
        );
        assert_eq!(shapes[5], (&[1, 4][..], &[1, 4][..]));
        assert_eq!(folded.output_shape, vec![1, 4]);
    }
}
