import argparse
import csv
import json
import os
import shutil
from typing import Callable

from PIL import Image


def scale_annotation(sf: float, path: str, callback: Callable[[list], None]) -> list[list]:
    with open(path, "r") as f:
        parsed = json.loads(f.read())
        for measure in parsed['system_measures']:
            callback([
                measure['left'] * sf, measure['top'] * sf, None, None,
                (measure['left'] + measure['width']) * sf,
                (measure['top'] + measure['height']) * sf,
                None, None
            ])


filename_blacklist = ["coco", "dataset", "all"]

if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("root")
    args = app.parse_args()

    root = os.path.normpath(args.root)
    print("Dataset root", root)

    target = os.path.join(root, "dataset")
    shutil.rmtree(target, ignore_errors=True)
    os.mkdir(target)

    filenames = os.listdir(path=root)
    total_items = 0
    for f in filenames:
        if f not in filename_blacklist:
            total_items += len(os.listdir(path=os.path.join(root, f, "img")))
    validate_threshold = total_items * 0.8

    writer = csv.writer(
        open(os.path.join(target, "annotations.csv"), "w"),
        quoting=csv.QUOTE_MINIMAL
    )
    written = 0

    for f in filenames:
        if f in filename_blacklist:
            continue

        piece_root = os.path.join(root, f)
        images = os.path.join(piece_root, "img")
        annotations = os.path.join(piece_root, "json")

        for image in os.listdir(path=images):
            annotation_path = os.path.join(
                annotations, image.split('.')[0] + ".json")

            def append_data(a):
                writer.writerow([
                    "TRAIN" if written < validate_threshold else "VALIDATION",
                    os.path.join(target, image), *a
                ])

            i = Image.open(os.path.join(images, image))
            i.load()

            if i.width < 600:
                sf = 600 / i.width
                i = i.resize((600, round(i.height * sf)),
                             resample=Image.BILINEAR)
                scale_annotation(sf, annotation_path, append_data)
            elif i.height < 600:
                sf = 600 / i.height
                i = i.resize((round(i.width * sf), 600),
                             resample=Image.BILINEAR)
                scale_annotation(sf, annotation_path, append_data)
            else:
                scale_annotation(1, annotation_path, append_data)

            i.save(os.path.join(target, image))
            written += 1
