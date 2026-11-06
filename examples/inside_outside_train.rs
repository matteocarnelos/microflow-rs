use microflow_odt_macros::model;
use nalgebra::{matrix, SMatrix};
use ndarray::Array3;
use ndarray_npy::{read_npy, ReadNpyError};
use rand::{rng, seq::SliceRandom};
use std::{array, fs::read_dir};
// #[model("models/train/sine.tflite", 1, "mse", false)]
// struct Sine {}
// #[model("models/train/lenet.tflite", 5, "crossentropy", true, [30000.0,30000.0,0.0], [4096.0,4096.0,1024.0])]
// #[model("models/train/lenet.tflite", 3, "crossentropy", true, [30000.0,0.0], [4096.0,1024.0])]
#[model("models/train/outside_inside.tflite", 3, "crossentropy", true, [10000.0, 0.0], [8000.0, 1024.0])]
// #[model("models/train/lenet.tflite", 2, "crossentropy", true, [0.0], [1024.0])]
struct InsideOutside {}

fn main() {
    let mut label_0: Vec<[SMatrix<[f32; 3], 32, 32>; 1]> =
        read_dir("datasets/inside_outside_dataset/label0")
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
                    array::from_fn(|k| *array.get((i, j, k)).unwrap())
                })]
            })
            .collect();
    label_0.shuffle(&mut rng());
    let valid = read_dir("datasets/inside_outside_dataset/label1")
        .unwrap()
        .filter(|el| el.is_ok())
        .map(|el| -> Result<Array3<f32>, ReadNpyError> {
            let path = el.unwrap().path();
            // println!("{}", path.to_str().unwrap());
            read_npy(path)
        })
        .filter(|el| el.is_ok())
        .count();
    println!("valid: {}", valid);
    let mut label_1: Vec<[SMatrix<[f32; 3], 32, 32>; 1]> =
        read_dir("datasets/inside_outside_dataset/label1")
            .unwrap()
            .filter(|el| el.is_ok())
            .map(|el| {
                let path = el.unwrap().path();
                // println!("{}", path.to_str().unwrap());
                let array: Array3<f32> = read_npy(path).unwrap();
                // println!(
                //     "{},{},{}",
                //     array.shape()[0],
                //     array.shape()[1],
                //     array.shape()[2],
                // );
                [SMatrix::from_fn(|i, j| {
                    array::from_fn(|k| *array.get((i, j, k)).unwrap())
                })]
            })
            .collect();
    label_1.shuffle(&mut rng());
    let validation_percentage = 0.2;
    let validation_0 =
        label_0.split_off((label_0.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let validation_1 =
        label_1.split_off((label_1.len() as f32 * (1f32 - validation_percentage)).round() as usize);
    let mut model = InsideOutside::new();
    let epochs = 25;
    let batch = 20;
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
        .weights0
        .buffer
        .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
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
        let initial = model.weights0.buffer.clone().cast::<i32>();
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
        let fin = model.weights0.buffer.cast::<i32>();
        let diff = fin - initial;
        let changed = diff
            .map(|el| if el != 0 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        let saturated = model
            .weights0
            .buffer
            .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        println!("saturated params {}", saturated);
        println!("changed params {}", changed);
        println!("validation accuracy : {}/{}", correct, validation_vec.len());
    }
}
