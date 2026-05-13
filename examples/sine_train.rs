use libm::{cosf, sinf, sqrtf};
use microflow_odt_macros::model;
use nalgebra::{matrix, SMatrix};
use rand::Rng;
use std::{fs::File, io::Write};
// #[model("models/train/sine.tflite", 3, "mse", false, [20000.0,40000.0,0.0],[256.0,1024.0,256.0])]
#[model("models/train/sine.tflite", 2, "mse", false, [40000.0,0.0],[1024.0,256.0])]
// #[model("models/train/sine.tflite", 1, "mse", false, [0.0],[256.0])] //50 epochs enough
struct Sine {}

fn main() {
    let mut rng = rand::rng();
    let output_scale = 0.00785118155181408;
    let output_zero_point = 1i8;
    let epochs = 400;
    let samples = 1000;
    let batch = 16;
    let learning_rate = 0.01;
    let mut errors: Vec<f32> = vec![];
    for _ in 0..5 {
        let mut model = Sine::new();
        let mut counter = 0;
        let initial = model.weights0.buffer.clone();
        println!("initial_weights: {}", initial);
        'epochs: for e in 0..epochs {
            println!("epoch {}", e);
            for sample in 0..samples {
                let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
                // let y = 0.5 * sinf(x);
                // let y = x / 4f32;
                // let y = cosf(x);
                let y = sinf(x) / x;
                // println!("x unquantized: {x}");
                // println!("y unquantized: {y}");
                let output = microflow::tensor::Tensor2D::quantize(
                    matrix![y],
                    [output_scale],
                    [output_zero_point],
                );
                // println!("y quantized: {}", output.buffer[0]);

                let y_p = model.predict_train(matrix![x], &output, learning_rate)[0];
                // println!("predicted: {y_p}");
                // println!(
                // "predicted quantized: {}",
                // microflow::quantize::quantize(y_p, output_scale, output_zero_point)
                // );
                // println!("batch back gradient: {}", model.weights0_gradient);
                // println!("++++++++++++++++++++++++++++");
                if (sample + 1) % batch == 0 {
                    // println!("batch back gradient: {}", model.weights0_gradient);

                    // microflow::update_layer::update_weights_perc_2D::<i8, 16, 1, 4>(
                    //     &mut model.weights0,
                    //     &model.weights0_gradient,
                    //     batch,
                    //     learning_rate,
                    // );
                    // println!("final_gradient: {}", model.weights0_gradient);
                    // println!("final_weights: {}", model.weights0.buffer);
                    // println!(
                    //     "model batch gradient = {}",
                    //     // learning_rate * model.weights0_gradient.cast::<f32>() / batch as f32
                    //     model.weights0_gradient
                    // );
                    // println!(
                    //     "final_weights difference: {}",
                    //     model.weights0.buffer.cast::<i32>() - initial.cast::<i32>()
                    // );
                    // if counter == 0 {
                    //     break 'epochs;
                    // }
                    counter += 1;
                    model.update_layers(batch as usize, learning_rate);
                    // panic!();
                }
            }
        }
        println!("final_weights: {}", model.weights0.buffer);
        println!(
            "final_weights difference: {}",
            model.weights0.buffer.cast::<i32>() - initial.cast::<i32>()
        );
        let mut output_file = File::create("output_sine_trained").unwrap();
        let mut error = 0f32;
        for _ in 0..samples {
            let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            // let y = 0.5 * sinf(x);
            // let y = cosf(x);
            let y = sinf(x) / x;
            let output = model.predict(matrix![x]);
            error += (output[0] - y).powi(2);
            let output_string = format!("{} {} {}\n", x, y, output[0]);
            output_file.write_all(output_string.as_bytes()).unwrap();
        }
        errors.push(sqrtf(error));
        println!("error: {}", sqrtf(error));
    }
    println!(
        "errors: {},{},{},{},{}",
        errors[0], errors[1], errors[2], errors[3], errors[4]
    );
}
