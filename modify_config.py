import argparse
import os

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

if __name__ == '__main__':
    app = argparse.ArgumentParser()

    app.add_argument("--pipeline_config", required=True)
    app.add_argument("--dataset_directory", required=True)
    app.add_argument("--mapping", default="mapping.pbtxt")

    args = app.parse_args()

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(args.pipeline_config, "r") as f:
        text_format.Parse(f.read(), pipeline_config)

    train_input = pipeline_config.train_input_reader
    eval_input = pipeline_config.eval_input_reader[0]

    train_reader = train_input.tf_record_input_reader
    eval_reader = eval_input.tf_record_input_reader

    train_reader.input_path[0] = os.path.normpath(os.path.join(
        args.dataset_directory, "all", "training.record-?????-of-00004"))

    eval_reader.input_path[0] = os.path.normpath(os.path.join(
        args.dataset_directory, "all", "validation.record-?????-of-00004"))

    train_input.label_map_path = args.mapping
    eval_input.label_map_path = args.mapping

    with open(args.pipeline_config, "w") as f:
        f.write(str(pipeline_config))
