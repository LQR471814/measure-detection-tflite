import argparse

import numpy as np
import tensorflow as tf
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("--dataset", help="directory to the dataset")
    app.add_argument("--model_dir", help="training data directory")
    args = app.parse_args()

    spec = model_spec.get('efficientdet_lite0')
    spec.config.tflite_max_detections = 50
    spec.config.model_dir = args.model_dir

    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(args.dataset)

    model = object_detector.create(
        train_data,
        validation_data=validation_data,
        model_spec=spec,
        epochs=1,
        batch_size=1,
        train_whole_model=True,
    )
    model.evaluate(test_data)

    model.export(export_dir='.')
    model.evaluate_tflite('model.tflite', test_data)
