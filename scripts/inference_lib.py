from glob import glob
from typing import Callable

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw

BoxList = list[tuple[int, int, int, int]]
NormalizedList = list[tuple[float, float, float, float]]
Runner = Callable[[np.array], NormalizedList]


def inference(
    runner: Runner,
    dimensions: tuple[int, int]
):
    width, height = dimensions
    for f in glob("examples/example_?.*"):
        example = Image.open(f).convert("RGB")
        example.load()
        # example_resized = example.resize((width, height), resample=Image.LINEAR)
        example = example.resize((width, height), resample=Image.LINEAR)
        example_np = np.expand_dims(np.array(example, dtype=np.uint8), 0)

        output = runner(example_np)
        boxes: BoxList = []

        for box in output:
            x1, y1 = box[1] * example.width, box[0] * example.height
            x2, y2 = box[3] * example.width, box[2] * example.height
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

        draw_bbox(example, boxes).save(
            f"{'.'.join(f.split('.')[:-1])}-bbox.png")


def draw_bbox(
    image: Image.Image,
    boxes: BoxList
) -> Image.Image:
    overlay = Image.new("RGBA", image.size)
    overlay_canvas = ImageDraw(overlay)

    for x1, y1, x2, y2 in boxes:
        overlay_canvas.rectangle(
            [x1, y1, x2, y2], fill='#00FFFF1B')
        overlay_canvas.rectangle(
            [x1, y1, x2, y2], outline='#008888', width=2)

    result_image = Image.alpha_composite(
        image.convert("RGBA"), overlay).convert("RGB")
    return result_image.convert("RGB")