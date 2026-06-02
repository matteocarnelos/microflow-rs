use microflow::activation::FusedActivation;
use microflow::ops::{
    add_2d, concatenation_4d, logistic_2d, max_pool_2d, mul_2d, pad_4d, tanh_2d, AddOptions,
    MaxPool2DOptions, MulOptions,
};
use microflow::tensor::{Tensor2D, Tensor4D, TensorViewPadding};
use nalgebra::{matrix, Const};

#[test]
fn add_mul_predict_paths() {
    let lhs = Tensor2D::quantize(matrix![0.5, 1.0], [0.25], [0i8]);
    let rhs = Tensor2D::quantize(matrix![0.5, 0.5], [0.25], [0i8]);

    let add = add_2d(
        lhs,
        &rhs,
        [0.25],
        [0],
        AddOptions {
            fused_activation: FusedActivation::None,
        },
    );
    let mul = mul_2d(
        add,
        &Tensor2D::quantize(matrix![0.5, 0.5], [0.25], [0i8]),
        [0.25],
        [0],
        MulOptions {
            fused_activation: FusedActivation::None,
        },
    );
    let output = mul.dequantize();
    assert_eq!(output[(0, 0)] <= output[(0, 1)], true);
}

#[test]
fn pooling_concat_pad_activation_paths() {
    let input = Tensor4D::new(
        [matrix![
            [1, 2], [3, 4];
            [5, 6], [7, 8]
        ]],
        [0.25],
        [0],
    );
    let pooled = max_pool_2d(
        input,
        (Const::<2>, Const::<2>),
        [0.25],
        [0],
        MaxPool2DOptions {
            fused_activation: FusedActivation::None,
            view_padding: TensorViewPadding::Valid,
            strides: (1, 1),
        },
    );
    let concat: Tensor4D<i8, 1, 1, 1, 4, 1> = concatenation_4d(
        pooled,
        &Tensor4D::new([matrix![[9, 10]]], [0.25], [0]),
        [0.25],
        [0],
    );
    let padded: Tensor4D<i8, 1, 1, 3, 4, 1> =
        pad_4d(concat, [(0, 0), (0, 0), (1, 1), (0, 0)], -1);
    assert_eq!(padded.buffer[0][(0, 0)][0], -1);
    assert_eq!(padded.buffer[0][(0, 2)][3], -1);
}

#[test]
fn logistic_tanh_quantized_paths() {
    let input = Tensor2D::new(matrix![1, -1], [0.5], [0]);
    let logistic = logistic_2d(input, [1.0 / 128.0], [0]);
    let tanh = tanh_2d(logistic, [1.0 / 128.0], [0]);
    assert_eq!(tanh.buffer[(0, 0)] > tanh.buffer[(0, 1)], true);
}
