use core::array;
use microflow::buffer::{Buffer2D, Buffer4D};
use microflow_odt_macros::model;
use nalgebra::{matrix, SMatrix};
use ndarray::Array3;
use ndarray_npy::read_npy;
use rand::{rng, seq::SliceRandom};
use std::fs::read_dir;
// #[model("models/train/sine.tflite", 1, "mse", false)]
// struct Sine {}
// #[model("models/train/lenet.tflite", 5, "crossentropy", true, [30000.0,30000.0,0.0], [8192.0,4096.0,1024.0])]
#[model("models/train/lenet.tflite", 5, "crossentropy", true, [30000.0,30000.0,0.0], [16000.0,4096.0,1024.0])]
struct LeNet {}

fn main() {
    let mut label_0: Vec<[SMatrix<[f32; 1], 28, 28>; 1]> =
        read_dir("datasets/fine_dataset_lenet/label_0")
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
                [SMatrix::from_fn(|i, j| [*(array.get((i, j, 0)).unwrap())])]
            })
            .collect();
    label_0.shuffle(&mut rng());
    let mut label_1: Vec<[SMatrix<[f32; 1], 28, 28>; 1]> =
        read_dir("datasets/fine_dataset_lenet/label_1")
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
                [SMatrix::from_fn(|i, j| [*(array.get((i, j, 0)).unwrap())])]
            })
            .collect();
    label_1.shuffle(&mut rng());
    let validation_percentage = 0.1;
    let validation_0 =
        label_0.split_off((label_0.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let validation_1 =
        label_1.split_off((label_1.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let mut model = LeNet::new();
    let epochs = 25;
    let batch = 50;
    let learning_rate = 0.01;
    let mut train_vec: Vec<_> = label_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(label_1.into_iter().map(|x| (x, 1)))
        .collect();
    train_vec.shuffle(&mut rng());
    train_vec = train_vec.drain(..2000).collect();
    let mut validation_vec: Vec<_> = validation_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(validation_1.into_iter().map(|x| (x, 1)))
        .collect();
    validation_vec.shuffle(&mut rng());
    let validation_vec: Vec<_> = validation_vec.drain(..200).collect();
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
            if index % batch == 0 {
                println!("batch: {}", index / batch);
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
        let diff: Buffer4D<i32, 120, 5, 5, 16> = core::array::from_fn(|batch| {
            SMatrix::from_fn(|i, j| {
                core::array::from_fn(|chan| fin[batch][(i, j)][chan] - initial[batch][(i, j)][chan])
            })
        });
        let changed = diff
            .iter()
            .flat_map(|batch| {
                batch
                    .iter()
                    .flat_map(|arr| arr.iter().map(|el| if *el != 0 { 1 } else { 0 }))
            })
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
        println!("validation accuracy : {}/{}", correct, validation_vec.len());
    }
}
