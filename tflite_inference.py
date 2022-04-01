import argparse

import tensorflow as tf
import numpy as np

from inference_lib import inference, Runner


def runner(func: tf.function) -> Runner:
    def wrapped(image: np.array):
        result = func(input=image)
        return result['output_3'][0]
    return wrapped


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("tflite", help="Path of the tflite model")
    args = app.parse_args()

    model = tf.lite.Interpreter(args.tflite)
    signature = model.get_signature_runner()

    _, height, width, __ = model.get_input_details()[0]['shape']

    inference(runner(signature), (width, height))
