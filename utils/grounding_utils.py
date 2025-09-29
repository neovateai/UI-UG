import cv2
import numpy as np
import requests
import json
from qwen_vl_utils import smart_resize

IMAGE_FACTOR = 28
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

def visualize_box_image(name, url, boxes_with_labels_ori, coord_parser="sr"):
    boxes_with_labels_ori = json.loads(boxes_with_labels_ori)

    boxes_with_labels = []
    for box in boxes_with_labels_ori:
        try:
            box_coord = [int(num.strip()) for pair in box["box"].strip("()").split("),(") for num in pair.split(",")]
        except:
            box_coord = json.loads(box["box"])

        boxes_with_labels.append((box_coord, box['type']))

    response = requests.get(url)
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    width = img.shape[1]
    height = img.shape[0]
    r_height, r_width = smart_resize(height, width, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS)

    box_cnt = -1
    for box, label in boxes_with_labels:
        box_cnt += 1
        x1, y1, x2, y2 = map(int, box)

        if coord_parser == "sr":
            x1 = int(x1 / r_width * width + 0.5)
            y1 = int(y1 / r_height * height + 0.5)
            x2 = int(x2 / r_width * width + 0.5)
            y2 = int(y2 / r_height * height + 0.5)
        elif coord_parser == "norm":
            x1 = round(x1 * width, 3)
            y1 = round(y1 * height, 3)
            x2 = round(x2 * width, 3)
            y2 = round(y2 * height, 3)
        elif coord_parser == "qwen2":
            x1 = int(x1 / 1000 * width + 0.5)
            y1 = int(y1 / 1000 * height + 0.5)
            x2 = int(x2 / 1000 * width + 0.5)
            y2 = int(y2 / 1000 * height + 0.5)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        padding = 3

        label_y = max(y1 - 10, text_height + padding)
        label_x = x1
        if label_x + text_width > img.shape[1]:
            label_x = img.shape[1] - text_width - 5
        cv2.putText(
            img,
            str(box_cnt) + "_" + label,
            (label_x + padding, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA
        )

    cv2.imwrite(f"{name}.png", img)