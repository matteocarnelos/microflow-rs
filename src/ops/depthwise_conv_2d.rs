use crate::activations::ActivationType;
use crate::tensor::Buffer4D;

// TODO: Support quantized version

pub enum PaddingType {
    SAME,
    VALID,
}

pub fn depthwise_conv_2d<
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    const D8: usize,
>(
    input: &Buffer4D<f32, D1, D2, D3, D4>,
    weights: Buffer4D<f32, D1, D5, D6, D4>,
    biases: [f32; D4],
    fused_activation: ActivationType,
    padding: PaddingType,
) -> Buffer4D<f32, D1, D7, D8, D4> {
    todo!()
}
