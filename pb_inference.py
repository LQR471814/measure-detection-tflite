import argparse

import tensorflow as tf
import numpy as np

from inference_lib import inference, Runner


def runner(func: tf.function) -> Runner:
    def wrapped(image: np.array):
        result = func(input_tensor=image.astype(np.uint8))
        print(result)
        return result['detection_boxes'][0]
    return wrapped


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("directory", help="Directory of the saved_model")
    args = app.parse_args()

    model = tf.saved_model.load(args.directory)
    inference(runner(model.signatures['serving_default']), (640, 640))
