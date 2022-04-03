PYTHON = py -3.9

OBJECT_DETECTION_LIB = models/research/object_detection
DATASET_DIRECTORY = d:/datasets/AudioLabs_v2
PIPELINE_CONFIG = configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config
INIT = $(OBJECT_DETECTION_LIB) $(PIPELINE_CONFIG)

TRAIN_DIR = data/train
PB_DIR = data/output_pb
TFLITE_DIR = data/output_tflite
TFLITE_MODEL = model.tflite

test: $(INIT)
	$(PYTHON) $(OBJECT_DETECTION_LIB)/builders/model_builder_test.py

prepare-dataset: $(INIT)
	$(PYTHON) create_tf_records.py \
		--label_map_path="mapping.pbtxt" \
		--included_classes="system_measures" \
		--image_directory="$(DATASET_DIRECTORY)/dataset" \
		--annotation_directory="$(DATASET_DIRECTORY)/dataset" \
		--output_path_training_split="$(DATASET_DIRECTORY)/all/training.record" \
		--output_path_validation_split="$(DATASET_DIRECTORY)/all/validation.record" \
		--output_path_test_split="$(DATASET_DIRECTORY)/all/test.record"

train: $(INIT)
	$(PYTHON) $(OBJECT_DETECTION_LIB)/model_main_tf2.py \
		--pipeline_config_path="$(PIPELINE_CONFIG)" \
		--model_dir="$(TRAIN_DIR)" \
		--alsologtostderr

evaluate: $(INIT) $(TRAIN_DIR)
	$(PYTHON) $(OBJECT_DETECTION_LIB)/model_main_tf2.py \
		--pipeline_config_path="$(PIPELINE_CONFIG)" \
		--model_dir="$(TRAIN_DIR)" \
		--checkpoint_dir="$(TRAIN_DIR)" \
		--alsologtostderr

freeze-pb: $(INIT) $(TRAIN_DIR)
	$(PYTHON) $(OBJECT_DETECTION_LIB)/exporter_main_v2.py \
		--input_type=image_tensor \
		--pipeline_config_path="$(PIPELINE_CONFIG)" \
		--trained_checkpoint_dir="$(TRAIN_DIR)" \
		--output_directory="$(PB_DIR)"

freeze-tflite: $(INIT) $(TRAIN_DIR)
	$(PYTHON) $(OBJECT_DETECTION_LIB)/export_tflite_graph_tf2.py \
		--pipeline_config_path="$(PIPELINE_CONFIG)" \
		--trained_checkpoint_dir="$(TRAIN_DIR)" \
		--output_directory="$(TFLITE_DIR)" \
		--max_detections 100
	tflite_convert \
		--saved_model_dir="$(TFLITE_DIR)/saved_model" \
		--output_file="$(TFLITE_MODEL)"

inference-pb: $(INIT) $(PB_DIR)
	$(PYTHON) pb_inference.py $(PB_DIR)/saved_model

inference-tflite: $(TFLITE_MODEL)
	$(PYTHON) tflite_inference.py $(TFLITE_MODEL)
