import tensorflow as tf
import numpy as np

def predict(interpreter, input):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    input = input / input_scale + input_zero_point
    input = input.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    output = output.astype(np.float32)
    output = (output - output_zero_point) * output_scale
    return output
