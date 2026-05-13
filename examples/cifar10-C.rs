use microflow_odt_macros::model;
// use microflow_macros::model;
use microflow::tensor::Tensor2D;
use microflow::tensor::Tensor4D;
use nalgebra::SMatrix;
use ndarray::Array3;
use ndarray_npy::read_npy;
use rand::{rng, seq::SliceRandom};
use std::fs::read_dir;
// #[model("models/train/cifar10_quantized_fixed_batch.tflite", 3, "crossentropy", true, [30000.0,0.0], [4096.0,1024.0])]
// #[model("models/train/cifar10_quantized_fixed_batch.tflite", 3, "crossentropy", true, [0.0], [1024.0])]
#[model("models/train/cifar10_quantized_fixed_batch.tflite", 3, "crossentropy", true, [30000.0,0.0], [4096.0,4096.0])]
struct Cifar10 {}

fn main() {
    let all_labels_train: Vec<Vec<[SMatrix<[f32; 3], 32, 32>; 1]>> = (0..10)
        .into_iter()
        .map(|label| {
            read_dir(format!("datasets/cifar10c_dataset/train/label-{}", label))
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
                .collect()
        })
        .collect();
    let all_labels_test: Vec<Vec<[SMatrix<[f32; 3], 32, 32>; 1]>> = (0..10)
        .into_iter()
        .map(|label| {
            read_dir(format!("datasets/cifar10c_dataset/test/label-{}", label))
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
                .collect()
        })
        .collect();
    let mut all_labels_train: Vec<([SMatrix<[f32; 3], 32, 32>; 1], usize)> = all_labels_train
        .iter()
        .enumerate()
        .flat_map(|(ind, el)| el.iter().map(move |sample| (*sample, ind)))
        .collect();
    let all_labels_test: Vec<([SMatrix<[f32; 3], 32, 32>; 1], usize)> = all_labels_test
        .iter()
        .enumerate()
        .flat_map(|(ind, el)| el.iter().map(move |sample| (*sample, ind)))
        .collect();
    all_labels_train.shuffle(&mut rng());
    let mut model = Cifar10::new();
    let epochs = 25;
    let batch = 50;
    let learning_rate = 0.01;
    let output_scale = 0.00390625;
    let output_zero_point = -128i8;
    println!("train elements {}", all_labels_train.len());

    // let saturated0 = model
    //     .weights0
    //     .buffer
    //     .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
    //     .fold(0, |acc, el| acc + el);
    let saturated1 = model
        .weights1
        .buffer
        .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
        .fold(0, |acc, el| acc + el);
    let correct = all_labels_test
        .iter()
        .map(|sample| {
            let result = model.predict(sample.0);
            let max = result.iamax_full();
            println!(
                "validation result: {},{},{},{},{},{},{},{},{},{}",
                result[0],
                result[1],
                result[2],
                result[3],
                result[4],
                result[5],
                result[6],
                result[7],
                result[8],
                result[9]
            );
            if sample.1 == max.1 {
                1
            } else {
                0
            }
        })
        .reduce(|acc, val| acc + val)
        .unwrap();
    println!("correct: {}/{}", correct, all_labels_test.len());
    // println!("saturated params0 initially {}", saturated0);
    println!("saturated params1 initially {}", saturated1);
    for epoch in 0..epochs {
        // let initial0 = model.weights0.buffer.clone().cast::<i32>();
        let initial1 = model.weights1.buffer.clone().cast::<i32>();
        all_labels_train.shuffle(&mut rng());
        for (index, sample) in all_labels_train.iter().enumerate() {
            let y: SMatrix<f32, 1, 10> =
                SMatrix::from_fn(|_, ind| if ind == sample.1 { 1f32 } else { 0f32 });
            let output =
                microflow::tensor::Tensor2D::quantize(y, [output_scale], [output_zero_point]);

            let predicted_output = model.predict_train(sample.0, &output, learning_rate);
            if index > 0 && index % batch == 0 {
                println!("batch: {}", index / batch);
                model.update_layers(batch, learning_rate);
                // println!("new bias: {}", model.constants0.0)
            }
        }
        model.update_layers(batch, learning_rate);
        let correct = all_labels_test
            .iter()
            .map(|sample| {
                let result = model.predict(sample.0);
                let max = result.iamax_full();
                if sample.1 == max.1 {
                    1
                } else {
                    0
                }
            })
            .reduce(|acc, val| acc + val)
            .unwrap();
        // let fin0 = model.weights0.buffer.cast::<i32>();
        // let diff0 = fin0 - initial0;
        let fin1 = model.weights1.buffer.cast::<i32>();
        let diff1 = fin1 - initial1;
        // let changed0 = diff0
        //     .map(|el| if el != 0 { 1 } else { 0 })
        //     .fold(0, |acc, el| acc + el);
        let changed1 = diff1
            .map(|el| if el != 0 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        // let saturated0 = model
        //     .weights0
        //     .buffer
        //     .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
        //     .fold(0, |acc, el| acc + el);
        let saturated1 = model
            .weights1
            .buffer
            .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        // println!("saturated params0 {}", saturated0);
        println!("saturated params1 {}", saturated1);
        // println!("changed params0 {}", changed0);
        println!(
            "changed params1 {}/{}",
            changed1,
            model.weights1.buffer.shape().0*
            model.weights1.buffer.shape().1
        );
        println!(
            "validation accuracy : {}/{}",
            correct,
            all_labels_test.len()
        );
        println!("epoch {} concluded", epoch);
    }
}
