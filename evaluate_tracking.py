from darknet import YoloDetector
from pathlib import Path
from BoxLibrary import *
from my_library import *
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from tqdm.contrib import tzip
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("image_dir", type=str,
        help="Directory where sequential images, ground truths, optical_flow and image_list are stored.")
    parser.add_argument("label", type=str,
        help="The label to track.")

    parser.add_argument("--conf_threshold", "-t", type=float, default=0.25,
        help="Network confidence threshold")
    parser.add_argument("--dist_threshold", "-d", type=float, default=0.05, 
        help="Distance threshold for the evaluation.")
    parser.add_argument("--tracker_dist_threshold", "-u", type=float, default=None,
        help="Distance threshold for tracking. If None, default value for the stem is used.")
    parser.add_argument("--min_dets", "-m", type=int, default=None,
        help="Minimum number of detection for a valid track. If None, default value for the stem is used.")

    parser.add_argument("--net_dir", "-n", type=str, default=None,
        help="Diretory where cfg, meta and weights are stored. If not specified, stored predictions will be used.")
    parser.add_argument("--compute_flow", "-f", action="store_true",
        help="Compute the optical flow on demand.")
    parser.add_argument("--save_flow", "-s", type=str, default=None,
        help="Save the computed optical flow to disk.")
    parser.add_argument("--save_dets", "-p", type=str, default=None,
        help="Save detections to disk if using network.")
    parser.add_argument("--annotation_ext", "-e", type=str, default = "xml", 
        help="The file format of ground truths. Either 'xml' or 'json'.")
    parser.add_argument("--generate_img_list", "-g", action="store_true",
        help="Generate the image list file.")
    parser.add_argument("--no_tracking", action="store_false", dest="do_tracking",
        help="Evaluate the raw detection without tracking.")

    args = parser.parse_args()

    args.image_dir = Path(args.image_dir).expanduser().resolve()
    args.optflow_path = args.image_dir / "optical_flow.txt"
    args.image_list_path = args.image_dir / "image_list.txt"
    args.dets_dir = args.image_dir / "predictions"

    valid_labels = {"maize", "bean", "leek", "stem_maize", "stem_bean", "stem_leek"}

    assert args.label in valid_labels, f"'{args.label}' is not a valid label"
    assert args.optflow_path.is_file() or args.compute_flow, "either the --compute_flow argument should be provided or the file 'optical_flow.txt' should be present in 'image_dir'"

    if args.tracker_dist_threshold is None:
        args.tracker_dist_threshold = (12 if "maize" in args.label else 6) / 100

    if args.min_dets is None:
        args.min_dets = 10 if "maize" in args.label else 13

    assert args.annotation_ext in {"xml", "json"}, f"'{args.annotation_ext}' is not a valid annotation extension (allowed: 'xml' and 'json')"

    return args


def associate(txt_file, flows, boxes):
    images = read_image_txt_file(txt_file)
    (img_width, img_height) = image_size(images[0])
    out_boxes = BoundingBoxes()

    for i, image in enumerate(images):
        image = str(Path(image).expanduser().resolve())
        (dx, dy) = OpticalFlow.traverse_backward(flows[:i+1], 0, 0)  # +1 !!!
        xmin = dx
        ymin = dy
        xmax = img_width + dx
        ymax = img_height + dy

        image_boxes = boxes.boxes_in((xmin, ymin, xmax, ymax))
        image_boxes = image_boxes.movedBy(-xmin, -ymin)

        for box in image_boxes:
            (x, y, w, h) = box.getAbsoluteBoundingBox()
            out_boxes.append(BoundingBox(imageName=image, classId=box.getClassId(), x=x, y=y, w=w, h=h, imgSize=box.getImageSize(), bbType=BBType.Detected, classConfidence=box.getConfidence()))

    return out_boxes


def track(boxes, image_list, flows, image_names, conf_thresh, min_points, dist_thresh, crop_percent=5/100):
    images = read_image_txt_file(image_list)
    images = [str(Path(p).resolve()) for p in images]
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)

    # Python dict are ordered but safer to iterate over image_names as key than dict.items()
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for image_name, flow in tzip(images, flows, desc="Tracking", unit="image"):
        tracker.update(boxes_by_name[image_name], flow)

    dets = tracker.get_filtered_boxes()
    dets = associate(image_list, flows, dets)
    dets = BoundingBoxes([det for det in dets
        if det.getImageName() in image_names
        and det.centerIsIn((crop_percent, crop_percent, 1.0 - crop_percent, 1.0 - crop_percent), as_percent=True)
    ])
    return dets, tracker


if __name__ == "__main__":
    args = parse_args()

    if args.generate_img_list:
        images = sorted(list(args.image_dir.glob("*.jpg")))
        args.image_list_path.write_text("\n".join(str(p.expanduser().resolve()) for p in images))

    en_to_fr = {
        "maize": "mais", "bean": "haricot", "leek": "poireau", 
        "stem_maize": "mais_tige", "stem_bean": "haricot_tige", "stem_leek": "poireau_tige"}
    fr_to_en = {v: k for k, v in en_to_fr.items()}

    if args.annotation_ext == "xml":
        gts = Parser.parse_xml_folder(args.image_dir, [en_to_fr[args.label]])
        gts.mapLabels(fr_to_en)
    else:  # json
        gts = Parser.parse_json_folder(args.image_dir, classes={args.label})

    if len(gts) == 0:
        print("WARNING: No annotation found. Maybe due to incorrect 'annotation_ext' parameter or no crop matching 'label' found.")

    if args.net_dir is None:
        boxes = Parser.parse_yolo_det_folder(args.dets_dir, args.image_dir, classes=[args.label])
    else:
        yolo = YoloDetector.from_dir(args.net_dir)
        boxes = BoundingBoxes()

        images = list(args.image_dir.glob("*.jpg")) if args.do_tracking else gts.getNames()

        for image in tqdm(images, desc="Detection"):
            img = np.array(Image.open(image))
            img_h, img_w = img.shape[:2]
            predictions = yolo.predict(img, args.conf_threshold)
            boxes += Parser.parse_yolo_darknet_detections(predictions, str(image.expanduser().resolve()), (img_w, img_h), classes=[args.label])
        
        if args.save_dets is not None:
            boxes.save(save_dir=args.save_dets)

    image_names = gts.getNames()
    if args.do_tracking:
        if args.compute_flow:
            flows = OpticalFlow.compute(args.image_list_path, mask_border=True, mask_egi=True)

            if args.save_flow is not None:
                OpticalFlow.save(flows, args.save_flow)
        else:
            flows = OpticalFlow.read(args.optflow_path)

        dets, _ = track(boxes, args.image_list_path, flows, image_names, args.conf_threshold, args.min_dets, args.tracker_dist_threshold)
    else:
        dets = BoundingBoxes([b for b in boxes if b.getImageName() in image_names])

    Evaluator().printF1ByClass(dets + gts, threshold=args.dist_threshold, method=EvaluationMethod.Distance)