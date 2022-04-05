PYTHON ?= py -3.9

DATASET_DIR ?= d:/datasets/AudioLabs_v2
MODEL_DIR ?= training_data

setup:
	$(PYTHON) -m pip install -q --use-deprecated=legacy-resolver tflite-model-maker
	$(PYTHON) -m pip install -q pycocotools

prepare-dataset:
	$(PYTHON) scripts/prepare_dataset.py $(DATASET_DIR)

model: $(DATASET_DIR)/dataset/annotations.csv
	$(PYTHON) main.py \
		--dataset="$(DATASET_DIR)/dataset/annotations.csv" \
		--model_dir="$(MODEL_DIR)"

inference:
	$(PYTHON) scripts/tflite_inference.py \
		--tflite="model.tflite"
