import os

import numpy as np
import tensorflow as tf
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig

if __name__ == '__main__':
    assert tf.__version__.startswith('2')
    tf.get_logger().setLevel('ERROR')
    from absl import logging
    logging.set_verbosity(logging.ERROR)

    spec = model_spec.get('efficientdet_lite0')
