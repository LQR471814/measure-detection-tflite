## measure-detection

***A measure detection model, based on the TFLite Model Maker***

This repository uses the `AudioLabs_v2` you can get it [here](https://github.com/apacha/OMR-Datasets/releases/download/datasets/AudioLabs_v2.zip)

### training

it is recommended that you have `make` installed to build the targets, however you can always run the scripts manually

if you want to change the command the `makefile` uses to run python, you can change the `PYTHON` environment variable.

before you start training you should ensure that the proper libraries are installed

```
pip install tensorflow numpy
make setup
```

then configure the path to the dataset with the `DATASET_DIR` environment variable and run

```
make dataset
```

we are now ready to begin training, you can do this in one line

```
make model
```

and verify the results with

```
make inference
```
