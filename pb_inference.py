import argparse

import tensorflow as tf
import numpy as np

from inference_lib import inference, Runner


def runner(func: tf.function) -> Runner:
    def wrapped(image: np.array):
        results = []
        output = func(input_tensor=image)
        for i, r in enumerate(output['detection_boxes'][0]):
            if output['detection_scores'][0][i] > 0.6:
                results.append(r)
        return results
    return wrapped


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("directory", help="Directory of the saved_model")
    args = app.parse_args()

    model = tf.saved_model.load(args.directory)
    inference(runner(model.signatures['serving_default']), (640, 640))
