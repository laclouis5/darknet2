from darknet import YoloDetector
from pathlib import Path
from BoxLibrary import *
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("image_dir", type=str,
        help="Directory where images are stored.")
    parser.add_argument("net_dir", type=str, 
        help="directory where cfg, meta and weights are stored.")
    parser.add_argument("save_dir", type=str,
        help="Directory where to store detections (darknet txt format).")
    
    parser.add_argument("--confidence_threshold", "-t", type=float, default=5/1000,
        help="Network confidence threshold.")
    parser.add_argument("--save_confidence", "-c", type=bool, default=True, 
        help="Wether or not to save the bounding box confidence in the annotation file.")

    args = parser.parse_args()

    args.image_dir = Path(args.image_dir).expanduser().resolve()
    args.save_dir = Path(args.save_dir).expanduser().resolve()

    return args


if __name__ == "__main__":
    args = parse_args()

    net = YoloDetector.from_dir(args.net_dir)
    images = list(Path(args.image_dir).glob("*.jpg"))
    boxes = BoundingBoxes()

    for image in tqdm(images):
        img = np.array(Image.open(image))
        img_h, img_w = img.shape[:2]

        predictions = net.predict(img, conf_threshold=args.confidence_threshold)
        boxes += Parser.parse_yolo_darknet_detections(predictions, str(image), (img_w, img_h))

    boxes.save(save_dir=args.save_dir, save_conf=args.save_confidence)