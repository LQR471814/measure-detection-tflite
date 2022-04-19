import argparse
import csv
import json
import os
import shutil
from typing import Callable

from PIL import Image


def replace_extension(name: str, extension: str) -> str:
    return '.'.join(name.split('.')[:-1]) + f'.{extension}'


def scale_annotation(sf: float, path: str, callback: Callable[[list], None]) -> list[list]:
    with open(path, "r") as f:
        parsed = json.loads(f.read())
        for measure in parsed['system_measures']:
            left = measure['left'] * sf
            top = measure['top'] * sf
            right = (measure['left'] + measure['width']) * sf
            bottom = (measure['top'] + measure['height']) * sf

            callback([left, top, right, bottom])


filename_blacklist = ["coco", "dataset", "all"]

if __name__ == '__main__':
    app = argparse.ArgumentParser()
    app.add_argument("root")
    app.add_argument("--ratio", type=int, default=1)
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
    total_items = total_items / args.ratio
    validate_threshold = total_items * 0.8
    test_threshold = total_items * 0.9

    writer = csv.writer(
        open(os.path.join(target, "annotations.csv"),
             "w", newline='', encoding='utf-8'),
        quoting=csv.QUOTE_MINIMAL
    )
    written = 0

    for f in filenames:
        if f in filename_blacklist:
            continue

        piece_root = os.path.join(root, f)
        images = os.path.join(piece_root, "img")
        annotations = os.path.join(piece_root, "json")

        for index, image in enumerate(os.listdir(path=images)):
            if not index % args.ratio == 0:
                continue

            annotation_path = os.path.join(
                annotations, replace_extension(image, "json"))

            i = Image.open(os.path.join(images, image))
            i.load()

            image = replace_extension(image, "jpg")

            def append_data(a):
                writer.writerow([
                    "TEST" if written >= test_threshold else
                    "VALIDATION" if written >= validate_threshold else
                    "TRAIN",
                    os.path.join(target, image),
                    "measure",
                    a[0] / i.width,
                    a[1] / i.height,
                    None, None,
                    a[2] / i.width,
                    a[3] / i.height,
                    None, None
                ])

            if i.width > 1024:
                sf = 1024 / i.width
                i = i.resize((1024, round(i.height * sf)),
                             resample=Image.BILINEAR)
                scale_annotation(sf, annotation_path, append_data)
            elif i.height > 1024:
                sf = 1024 / i.height
                i = i.resize((round(i.width * sf), 1024),
                             resample=Image.BILINEAR)
                scale_annotation(sf, annotation_path, append_data)
            else:
                scale_annotation(1, annotation_path, append_data)

            i.save(os.path.join(target, image))
            written += 1
