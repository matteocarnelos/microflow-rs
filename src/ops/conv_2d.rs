// use crate::activation::FusedActivation;
// use crate::tensor::Tensor4D;
//
// pub struct Conv2DOptions {
//     pub fused_activation: FusedActivation,
//     pub padding: Conv2DPadding,
//     pub strides: (usize, usize),
// }
//
// pub enum Conv2DPadding {
//     SAME,
//     VALID,
// }
//
// pub fn conv_2d<
//     T: Quantized,
//     const D1: usize,
//     const D2: usize,
//     const D3: usize,
//     const D4: usize,
//     const D4_OR_1: usize,
//     const D5: usize,
//     const D6: usize,
//     const D7: usize,
//     const D8: usize,
// >(
//     input: Tensor4D<T, 1, D2, D3, D4_OR_1, D4_OR_1>,
//     filters: Tensor4D<T, D1, D5, D6, D4, D4_OR_1>
// )
