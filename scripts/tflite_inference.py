import argparse

import tensorflow as tf
import numpy as np

from inference_lib import inference, Runner


def runner(func: tf.function) -> Runner:
    def wrapped(image: np.array):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)

        results = []
        output = func(images=image)

        boxes = output['output_3'][0]
        scores = output['output_1'][0]

        for i, r in enumerate(boxes):
            if scores[i] > 0.4:
                results.append(r)

        return results
    return wrapped


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("tflite", help="Path of the tflite model")
    args = app.parse_args()

    model = tf.lite.Interpreter(args.tflite)
    signature = model.get_signature_runner()
    _, height, width, __ = model.get_input_details()[0]['shape']
    inference(runner(signature), (width, height))
