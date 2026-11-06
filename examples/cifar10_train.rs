use core::array;
use microflow::buffer::{Buffer2D, Buffer4D};
// use microflow_odt_macros::model;
use microflow_odt_macros::model;
use nalgebra::{matrix, SMatrix};
use ndarray::Array3;
use ndarray_npy::read_npy;
use rand::{rng, seq::SliceRandom};
use std::fs::read_dir;
// #[model("models/train/sine.tflite", 1, "mse", false)]
// struct Sine {}
// #[model("models/train/lenet.tflite", 5, "crossentropy", true, [30000.0,30000.0,0.0], [8192.0,4096.0,1024.0])]
#[model("models/train/cifar10.tflite", 5, "crossentropy", true, [30000.0,30000.0,0.0], [16000.0,4096.0,4096.0])]
struct Cifar10 {}

fn main() {
    let mut label_0: Vec<[SMatrix<[f32; 3], 32, 32>; 1]> =
        read_dir("datasets/cifar10_dataset/label0")
            .unwrap()
            .filter(|el| el.is_ok())
            .map(|el| {
                let array: Array3<f32> = read_npy(el.unwrap().path()).unwrap();
                // println!(
                //     "{},{},{}",
                //     array.shape()[0],
                //     array.shape()[1],
                //     array.shape()[2],
                // );
                [SMatrix::from_fn(|i, j| {
                    [
                        *(array.get((i, j, 0)).unwrap()),
                        *(array.get((i, j, 1)).unwrap()),
                        *(array.get((i, j, 2)).unwrap()),
                    ]
                })]
            })
            .collect();
    label_0.shuffle(&mut rng());
    let mut label_1: Vec<[SMatrix<[f32; 3], 32, 32>; 1]> =
        read_dir("datasets/cifar10_dataset/label0")
            .unwrap()
            .filter(|el| el.is_ok())
            .map(|el| {
                let array: Array3<f32> = read_npy(el.unwrap().path()).unwrap();
                // println!(
                //     "{},{},{}",
                //     array.shape()[0],
                //     array.shape()[1],
                //     array.shape()[2],
                // );
                [SMatrix::from_fn(|i, j| {
                    [
                        *(array.get((i, j, 0)).unwrap()),
                        *(array.get((i, j, 1)).unwrap()),
                        *(array.get((i, j, 2)).unwrap()),
                    ]
                })]
            })
            .collect();
    label_1.shuffle(&mut rng());
    let validation_percentage = 0.1;
    let validation_0 =
        label_0.split_off((label_0.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let validation_1 =
        label_1.split_off((label_1.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let mut model = Cifar10::new();
    let constants_last = model.constants3;
    println!(
        "constants last layer {}, {}, {}, {}",
        constants_last.0, constants_last.1, constants_last.2, constants_last.3
    );
    let epochs = 25;
    let batch = 64;
    let learning_rate = 0.01;
    let mut train_vec: Vec<_> = label_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(label_1.into_iter().map(|x| (x, 1)))
        .collect();
    train_vec.shuffle(&mut rng());
    // train_vec = train_vec.drain(..2000).collect();
    let mut validation_vec: Vec<_> = validation_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(validation_1.into_iter().map(|x| (x, 1)))
        .collect();
    validation_vec.shuffle(&mut rng());
    // let validation_vec: Vec<_> = validation_vec.drain(..200).collect();
    let output_scale = 0.00390625;
    let output_zero_point = -128i8;
    println!("train elements {}", train_vec.len());
    println!(
        "validation split baseline {}/{}",
        &validation_vec
            .iter()
            .map(|el| el.1)
            .fold(0, |acc, el| acc + el),
        validation_vec.len()
    );
    let saturated = model
        .filters0
        .buffer
        .iter()
        .flat_map(|batch| {
            batch.iter().flat_map(|chans| {
                chans
                    .iter()
                    .map(|el| if *el >= 126 || *el <= -126 { 1 } else { 0 })
            })
        })
        .fold(0, |acc, el| acc + el);
    let correct = validation_vec
        .iter()
        .map(|sample| {
            let result = model.predict(sample.0);
            println!("validation result: {},{}", result[0], result[1]);
            if sample.1 == 1 && result[1] > result[0] {
                1
            } else if sample.1 == 0 && result[0] > result[1] {
                1
            } else {
                0
            }
        })
        .reduce(|acc, val| acc + val)
        .unwrap();
    println!("correct: {}", correct);
    println!("saturated params initially {}", saturated);
    for _ in 0..epochs {
        let initial = model
            .filters0
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        let initial1 = model
            .filters1
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        let initial_last = model.weights3.buffer.clone().map(|el| el as i32);
        train_vec.shuffle(&mut rng());
        for (index, sample) in train_vec.iter().enumerate() {
            let y = if sample.1 == 0 {
                matrix![1f32, 0f32]
            } else {
                matrix![0f32, 1f32]
            };
            let output =
                microflow::tensor::Tensor2D::quantize(y, [output_scale], [output_zero_point]);

            // println!("output: {}", output.buffer);
            let predicted_output = model.predict_train(sample.0, &output, learning_rate);
            // println!(
            //     "predicted output: {}",
            //     microflow::tensor::Tensor2D::quantize(
            //         predicted_output,
            //         [output_scale],
            //         [output_zero_point]
            //     )
            //     .buffer
            // );
            // println!("gradient: {}", model.weights0_gradient.view((0, 0), (4, 2)));
            // panic!();
            if index != 0 && (index + 1) % batch == 0 {
                if (index / batch) % 10 == 0 {
                    println!("batch: {}", index / batch);
                }
                model.update_layers(batch, learning_rate);
                // println!("new bias: {}", model.constants0.0)
            }
        }
        model.update_layers(batch, learning_rate);
        let correct = validation_vec
            .iter()
            .map(|sample| {
                let result = model.predict(sample.0);
                // println!("result: {}, {}", result[0], result[1]);
                if sample.1 == 1 && result[1] > result[0] {
                    1
                } else if sample.1 == 0 && result[0] > result[1] {
                    1
                } else {
                    0
                }
            })
            .reduce(|acc, val| acc + val)
            .unwrap();
        let fin = model
            .filters0
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        let fin1 = model
            .filters1
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        let fin_last = model.weights3.buffer.clone().map(|el| el as i32);
        let diff: Buffer4D<i32, 16, 3, 3, 16> = core::array::from_fn(|batch| {
            SMatrix::from_fn(|i, j| {
                core::array::from_fn(|chan| fin[batch][(i, j)][chan] - initial[batch][(i, j)][chan])
            })
        });
        let diff1: Buffer4D<i32, 16, 3, 3, 16> = core::array::from_fn(|batch| {
            SMatrix::from_fn(|i, j| {
                core::array::from_fn(|chan| {
                    fin1[batch][(i, j)][chan] - initial1[batch][(i, j)][chan]
                })
            })
        });
        let diff_last: Buffer2D<i32, 1024, 2> =
            SMatrix::from_fn(|i, j| fin_last[(i, j)] - initial_last[(i, j)]);
        let changed = diff
            .iter()
            .flat_map(|batch| {
                batch
                    .iter()
                    .flat_map(|arr| arr.iter().map(|el| if *el != 0 { 1 } else { 0 }))
            })
            .fold(0, |acc, el| acc + el);
        let changed = diff1
            .iter()
            .flat_map(|batch| {
                batch
                    .iter()
                    .flat_map(|arr| arr.iter().map(|el| if *el != 0 { 1 } else { 0 }))
            })
            .fold(0, |acc, el| acc + el);
        let changed_last = diff_last
            .iter()
            .map(|el| if *el != 0 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        let saturated = model
            .filters0
            .buffer
            .iter()
            .flat_map(|batch| {
                batch.iter().flat_map(|arr| {
                    arr.iter()
                        .map(|el| if *el >= 126 || *el <= -126 { 1 } else { 0 })
                })
            })
            .fold(0, |acc, el| acc + el);
        println!("saturated params {}", saturated);
        println!(
            "changed params {}/{}",
            changed,
            diff.len() * diff[0].shape().0 * diff[0].shape().0 * diff[0][0].len()
        );
        println!(
            "changed params1 {}/{}",
            changed,
            diff.len() * diff[0].shape().0 * diff[0].shape().0 * diff[0][0].len()
        );
        println!(
            "changed params last {}/{}",
            changed_last,
            diff_last.shape().0 * diff_last.shape().1
        );
        println!("validation accuracy : {}/{}", correct, validation_vec.len());
    }
}
