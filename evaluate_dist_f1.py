from darknet import YoloDetector
from BoxLibrary import *
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("image_dir", type=str,
        help="Directory where are stored ground truths and images.")
    parser.add_argument("--net_dir", "-n", type=str, default=None,
        help="Directory where cfg, meta and weights files are stored. If not provided, \
        'dets_dir' should be used.")
    parser.add_argument("--dets_dir", "-c", type=str, default=None,
        help="Directory where are stored detections. If not provided, \
        detections are done with the yolo network.")
    parser.add_argument("--dets_format", "-f", type=str, default="yolo",
        help="The file format of stored detection annotations.")
    parser.add_argument("--gts_format", "-g", type=str, default="yolo",
        help="the file format of stored ground truth annotations.")

    parser.add_argument("--conf_threshold", "-t", type=float, default=0.25,
        help="Confidence threshold for detector.")
    parser.add_argument("--dist_threshold", "-d", type=float, default=0.05,
        help="Distance threshold used for the F1 metric.")

    parser.add_argument("--save_csv", dest="csv_path", type=Path, default=None,
        help="Store the evaluation as csv.")

    args = parser.parse_args()

    assert args.net_dir is not None or args.dets_dir is not None
    assert args.dets_format in {"yolo", "json"}
    assert args.gts_format in {"yolo", "xml", "json"}

    return args


if __name__ == "__main__":
    args = parse_args()

    en_to_fr = {
        "maize": "mais", "bean": "haricot", "leek": "poireau", 
        "stem_maize": "mais_tige", "stem_bean": "haricot_tige", "stem_leek": "poireau_tige"}
    fr_to_en = {v: k for k, v in en_to_fr.items()}

    image_dir = Path(args.image_dir).expanduser().resolve()

    if args.dets_dir is not None:
        if args.dets_format == "yolo":
            boxes = Parser.parse_yolo_det_folder(args.dets_dir, image_dir)
        else:  # json
            boxes = Parser.parse_json_folder(args.dets_dir)
        boxes = BoundingBoxes([b for b in boxes if b.getConfidence() > args.conf_threshold])
        names = list(en_to_fr.keys())
    else:
        yolo = YoloDetector.from_dir(Path(args.net_dir).expanduser().resolve())
        names = yolo.class_names
        boxes = BoundingBoxes()
        images = list(image_dir.glob("*.jpg"))
        
        for image in tqdm(images):
            img = np.array(Image.open(image))
            img_h, img_w = img.shape[:2]
            predictions = yolo.predict(img, args.conf_threshold)

            boxes += Parser.parse_yolo_darknet_detections(predictions, str(image), (img_w, img_h))

    int_to_name = {str(i): n for i, n in enumerate(names)}

    if args.gts_format == "yolo":
        gts = Parser.parse_yolo_gt_folder(image_dir)
        gts.mapLabels(int_to_name)
    elif args.gts_format == "xml":
        gts = Parser.parse_xml_folder(image_dir)
        gts.mapLabels(fr_to_en)
    else:  # json
        gts = Parser.parse_json_folder(image_dir)

    Evaluator().printF1ByClass(boxes + gts, 
        threshold=args.dist_threshold, 
        method=EvaluationMethod.Distance, save_path=args.csv_path)