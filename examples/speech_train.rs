use libm::sinf;
use microflow::buffer::Buffer4D;
use microflow_odt_macros::model;
use nalgebra::{base::Matrix, base::SMatrix, matrix};
use ndarray::{Array3, Array4};
use ndarray_npy::read_npy;
use rand::{rng, seq::SliceRandom, Rng};
use std::fs::{read_dir, File};
// #[model("models/train/sine.tflite", 1, "mse", false)]
// struct Sine {}
// #[model("models/train/lenet.tflite", 1, "mse", false)]
// struct LeNet {}
#[model("models/train/speech_small_softmax.tflite", 4, "crossentropy", true, [80000.0,0.0], [8192.0,8192.0])]
// #[model("models/train/speech_small_softmax.tflite", 2, "crossentropy", true, [0.0], [8192.0])]
struct Speech {}

fn main() {
    let mut label_0: Vec<[SMatrix<[f32; 1], 125, 17>; 1]> =
        read_dir("datasets/dataset_fine_speech/label_0")
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
    let mut label_1: Vec<[SMatrix<[f32; 1], 125, 17>; 1]> =
        read_dir("datasets/dataset_fine_speech/label_1")
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
    let mut model = Speech::new();
    let epochs = 25;
    let batch = 100;
    let learning_rate = 0.01;
    let mut train_vec: Vec<_> = label_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(label_1.into_iter().map(|x| (x, 1)))
        .collect();
    let validation_vec: Vec<_> = validation_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(validation_1.into_iter().map(|x| (x, 1)))
        .collect();
    let output_scale = 0.00390625;
    let output_zero_point = -128i8;
    println!("train elements {}", train_vec.len());
    println!(
        "validation split {}/{}",
        validation_vec.iter().filter(|el| el.1 == 0).count(),
        validation_vec.len()
    );
    // let saturated = model
    //     .weights0
    //     .buffer
    //     .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
    //     .fold(0, |acc, el| acc + el);
    let saturated = model
        .weights0
        .buffer
        .iter()
        .flat_map(|batch| {
            batch.iter().flat_map(|arr| {
                arr.iter()
                    .map(|el| if *el >= 126 || *el <= -126 { 1 } else { 0 })
            })
        })
        .fold(0, |acc, el| acc + el);
    println!("saturated params initially {}", saturated);
    let correct = validation_vec
        .iter()
        .map(|sample| {
            let result = model.predict(sample.0);

            let exps: Vec<f32> = result.iter().map(|&x| (x).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let softm: Vec<f32> = exps.iter().map(|&x| x / sum).collect();
            println!("validation result, {},{}", softm[0], softm[1]);
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
    println!("validation accuracy : {}/{}", correct, validation_vec.len());
    for _ in 0..epochs {
        // let initial = model.weights0.buffer.clone().cast::<i32>();
        let initial = model
            .weights0
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        train_vec.shuffle(&mut rng());
        for (index, sample) in train_vec.iter().enumerate() {
            // println!("expected_output: {}", sample.1);
            let y = if sample.1 == 0 {
                matrix![1f32, 0f32]
            } else {
                matrix![0f32, 1f32]
            };
            let output =
                microflow::tensor::Tensor2D::quantize(y, [output_scale], [output_zero_point]);

            let out = model.predict_train(sample.0, &output, learning_rate);
            // println!(
            //     "output net: {}",
            //     microflow::tensor::Tensor2D::quantize(out, [output_scale], [output_zero_point])
            //         .buffer
            // );
            // println!("gradient: {}", model.weights0_gradient.view((0, 0), (4, 2)));
            // panic!();
            if index % batch == 0 {
                // println!(
                //     "final_gradient: {}",
                //     model.weights0_gradient.iter().fold(0i32, |meh, moh| meh
                //         + moh.fold(0i32, |acc, el| acc
                //             + el.iter().fold(0i32, |acc1, el1| acc1 + el1)))
                // );
                println!("batch {}", index / batch);
                model.update_layers(batch, learning_rate);
            }
        }
        model.update_layers(batch, learning_rate);
        let correct = validation_vec
            .iter()
            .map(|sample| {
                let result = model.predict(sample.0);

                let exps: Vec<f32> = result.iter().map(|&x| (x).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let softm: Vec<f32> = exps.iter().map(|&x| x / sum).collect();
                println!("validation result, {},{}", softm[0], softm[1]);
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
        // let fin = model.weights0.buffer.cast::<i32>();
        let fin = model
            .weights0
            .buffer
            .clone()
            .map(|batch| batch.map(|arr| arr.map(|el| el as i32)));
        let diff: Buffer4D<i32, 1, 10, 8, 8> = core::array::from_fn(|batch| {
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
            .weights0
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
