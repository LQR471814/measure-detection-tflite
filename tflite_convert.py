import tensorflow as tf
import argparse

if __name__ == "__main__":
    app = argparse.ArgumentParser()
    app.add_argument("model_path", help="Directory of the saved_model")
    args = app.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.model_path)
    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
