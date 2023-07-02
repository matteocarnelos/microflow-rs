# MicroFlow
> A [Rust](https://www.rust-lang.org) TinyML Compiler for Neural Network Inference on Embedded Systems

[![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)

MicroFlow is a robust and efficient TinyML inference engine designed for deploying machine learning models on embedded systems.
It was developed by Matteo Carnelos as part of his master's thesis project at the [University of Padova](https://www.unipd.it) in collaboration with [Grepit AB](https://www.grepit.se).

MicroFlow uses a compiler-based approach, resulting in the following engine structure:

<p align="center">
  <br/>
  <img src="res/structure-overview.svg" alt="structure-overview">
</p>

MicroFlow consists of two primary components: the compiler, represented by the `microflow-macros` crate, and the runtime, represented by the `microflow` crate.
The compiler, which runs prior to the Rust compiler, is responsible for parsing and pre-processing the model.
It generates the necessary source code to enable inference on the model.
On the other hand, the runtime is a `[no_std]` component designed to run on the target MCU.
It encompasses the implementation of operators, activation functions, and quantization procedures.

## Usage

MicroFlow utilizes Rust [Procedural Macros](https://doc.rust-lang.org/reference/procedural-macros.html) as its user interface.
By applying the `model` macro to a `struct` and providing the model's path, the MicroFlow compiler generates a `predict()` method.
This method can be called to perform inference on the given model.
Currently, MicroFlow only supports models in the TensorFlow Lite format (`.tflite`).

Here is a minimal example showcasing the usage of MicroFlow:

```rust ignore
use microflow::model;

#[model("path/to/model.tflite")]
struct MyModel;

fn main() {
    let prediction = MyModel::predict(input_data);
}
```

## Supported Operators

Currently, MicroFlow supports the following operators and activation functions:

| Operator          | Quantized | Tensor Type            |
|-------------------|-----------|------------------------|
| `FullyConnected`  | ✅         | `Tensor2D`             |
| `Conv2D`          | ✅         | `Tensor4D`             |
| `DepthwiseConv2D` | ✅         | `Tensor4D`             |
| `AveragePool2D`   | ✅         | `Tensor4D`             |
| `Reshape`         | ✅         | `Tensor2D`, `Tensor4D` |

| Activation Function | Quantized |
|---------------------|-----------|
| `ReLU`              | ✅         |
| `ReLU6`             | ✅         |
| `Softmax`           | ✅         |

These operators and activation functions cover common building blocks for neural networks and enable efficient inference with reduced memory and computational requirements.
However, MicroFlow's development roadmap includes plans for implementing additional operators and activation functions to expand the range of supported models.

## Tested Models and MCUs

The `examples` folder contains the code used to test MicroFlow on different MCUs.
These MCUs include:

- ESP32 (32-bit Xtensa)
- ATSAMV71 (32-bit Cortex-M7F)
- nRF52840 (32-bit Cortex-M4F)
- LM3S6965 (32-bit Cortex-M3)
- ATmega328 (8-bit AVR)

The models ued to test the inference engines can be found in the `models` directory.
These models include:

- A sine predictor
- A speech command recognizer (TinyConv)
- A person detector (MobileNet v1)

## Contributing

Contributors are welcome.
For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
