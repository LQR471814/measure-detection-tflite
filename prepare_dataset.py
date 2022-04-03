import argparse
import json
import os
import shutil
from google.protobuf import text_format

from PIL import Image


def scale_annotation(sf: float, path: str, name: str) -> str:
    with open(os.path.join(path, name), "r") as f:
        parsed = json.loads(f.read())

        parsed['width'] *= sf
        parsed['height'] *= sf

        for measure in parsed['system_measures']:
            measure["left"] *= sf
            measure["top"] *= sf
            measure["width"] *= sf
            measure["height"] *= sf

        # ? Remove unnecessary data
        parsed['stave_measures'] = []
        parsed['staves'] = []

        return json.dumps(parsed)


if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("root")
    args = app.parse_args()

    root = args.root
    print("Dataset root", root)

    target = os.path.join(root, "dataset")
    shutil.rmtree(target, ignore_errors=True)
    if not os.path.isdir(target):
        os.mkdir(target)

    for f in os.listdir(path=root):
        if f == "prepare_dataset.py" or f == "coco" or f == "dataset" or f == "all":
            continue

        piece_root = os.path.join(root, f)
        images = os.path.join(piece_root, "img")
        annotations = os.path.join(piece_root, "json")

        for image in os.listdir(path=images):
            annotation = image.split('.')[0] + ".json"
            annotated = ""

            path = os.path.join(images, image)
            i = Image.open(path)
            i.load()

            if i.width < 600:
                sf = 600 / i.width
                i = i.resize((600, round(i.height * sf)),
                             resample=Image.NEAREST)
                annotated = scale_annotation(sf, annotations, annotation)
            elif i.height < 600:
                sf = 600 / i.height
                i = i.resize((round(i.width * sf), 600),
                             resample=Image.NEAREST)
                annotated = scale_annotation(sf, annotations, annotation)
            else:
                annotated = scale_annotation(1, annotations, annotation)

            with open(os.path.join(target, annotation), "w") as f:
                f.write(annotated)
            i.save(os.path.join(target, image))
