PYTHON ?= py -3.9

DATASET_DIR ?= d:/datasets/AudioLabs_v2

setup:
	$(PYTHON) -m pip install -q --use-deprecated=legacy-resolver tflite-model-maker
	$(PYTHON) -m pip install -q pycocotools

prepare-dataset:
	$(PYTHON) scripts/prepare_dataset.py $(DATASET_DIR)
