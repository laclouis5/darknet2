import shutil
from pathlib import Path
from BoxLibrary import *
from my_library import *
from my_xml_toolbox import *
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from tqdm.contrib import tzip
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import shutil


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("image_dir", type=str,
        help="Directory where sequential images, ground truths, optical_flow and image_list are stored.")
    parser.add_argument("label", type=str,
        help="The label to track. Either 'mais' or 'haricot'.")

    parser.add_argument("--conf_threshold", "-t", type=float, default=0.25,
        help="Network confidence threshold")
    parser.add_argument("--dist_threshold", "-d", type=float, default=None,
        help="Distance threshold for tracking. If None, default value for the stem is used.")
    parser.add_argument("--min_dets", "-m", type=int, default=None,
        help="Minimum number of detection for a valid track. If None, default value for the stem is used.")

    parser.add_argument("--net_dir", "-n", type=str, default=None,
        help="Diretory where Yolo cfg, meta and weights are stored. If not specified, \
        predictions stored in 'predictions' will be used.")
    parser.add_argument("--save_dets", "-p", action="store_true",
        help="Save detections to disk if using network inference.")
    parser.add_argument("--dets_fmt", "-f", type=str, default = "yolo", 
        help="The file format of ground truths. Either 'xml', 'json' or 'yolo'.")
        
    parser.add_argument("--compute_flow", "-c", action="store_true",
        help="Compute the optical flow on demand.")
    parser.add_argument("--save_flow", "-s", action="store_true",
        help="Save the computed optical flow to disk.")

    parser.add_argument("--no_tracking", action="store_false", dest="do_tracking", 
        help="Disable tracking.")

    args = parser.parse_args()

    args.image_dir = Path(args.image_dir).expanduser().resolve()
    args.optflow_path = args.image_dir / "optical_flow.txt"
    args.image_list_path = args.image_dir / "image_list.txt"
    args.dets_dir = args.image_dir / "predictions"

    # valid_labels = {"maize", "bean", "leek", "stem_maize", "stem_bean", "stem_leek"}

    assert args.optflow_path.is_file() or args.compute_flow, "either the --compute_flow argument should be provided or the file 'optical_flow.txt' should be present in 'image_dir'"
    assert args.dets_dir.is_dir() or args.net_dir is not None, "either the network path should be provided with --net_dir for inference or detections should be provided with --dets_dir"

    if args.dist_threshold is None:
        args.dist_threshold = (12 if args.label == "mais" else 6) / 100

    if args.min_dets is None:
        args.min_dets = 10 if args.label == "mais" else 13

    if args.label == "mais":
        args.tracking_label = "stem_maize"
    elif args.label == "haricot":
        args.tracking_label = "stem_bean"
    else:
        raise AssertionError(f"'{args.label}' is not a valid label (either 'mais' or 'haricot')")

    assert args.dets_fmt in {"json", "yolo"}

    return args


def associate(txt_file, flows, boxes):
    images = read_image_txt_file(txt_file)
    (img_width, img_height) = image_size(images[0])
    out_boxes = BoundingBoxes()

    for i, image in enumerate(images):
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


def track(boxes, image_list, flows, conf_thresh, min_points, dist_thresh, crop_percent=3/100):
    images = read_image_txt_file(image_list)
    images = [str(Path(p).expanduser().resolve()) for p in images]
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)

    # Python dict are ordered but safer to iterate over image_names as key than dict.items()
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for image_name, flow in tzip(images, flows, desc="Tracking", unit="image"):
        tracker.update(boxes_by_name[image_name], flow)

    dets = tracker.get_filtered_boxes()
    dets = associate(image_list, flows, dets)
    dets = BoundingBoxes([det for det in dets
        if det.centerIsIn((crop_percent, crop_percent, 1.0 - crop_percent, 1.0 - crop_percent), as_percent=True)
    ])
    return dets


def inner(element):
        ((image, image_boxes), save_dir, label_name) = element

        image_name = Path(image).name
        new_image_name = f"bipbip_{label_name}{image_name[3:]}"
        shutil.copy(image, save_dir / new_image_name)
        (img_h, img_w) = cv.imread(image).shape[:2]
        radius = int(5/100 * min(img_w, img_h) / 2)

        xml_tree = XMLTree(new_image_name, width=img_w, height=img_h)

        for box in image_boxes:
            xml_tree.add_mask(label_name)

            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            rect = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]

            out_name = f"{Path(new_image_name).stem}_{xml_tree.plant_count-1}.png"

            stem_mask = Image.new(mode="1", size=(img_w, img_h))
            stem_mask.paste(Image.new(mode="1", size=(radius*2, radius*2), color=1), rect)
            stem_mask.save(os.path.join(save_dir, out_name))

            xml_tree.save(save_dir)


def operose(boxes, save_dir, label_name):
    create_dir(save_dir)
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    elements = list(zip(
        boxes_by_name.items(), 
        [save_dir]*len(boxes_by_name), 
        [label_name]*len(boxes_by_name)
    ))
    process_map(inner, elements, max_workers=4, desc="Operose", unit="image")


if __name__ == "__main__":
    args = parse_args()

    if not args.image_list_path.is_file():
        images = sorted(list(args.image_dir.glob("*.jpg")))
        args.image_list_path.write_text("\n".join(str(p.expanduser().resolve()) for p in images))

    if args.net_dir is None:
        if args.dets_fmt == "yolo":
            dets = Parser.parse_yolo_det_folder(args.dets_dir, args.image_dir, classes=[args.tracking_label])
        else:  # json
            dets = Parser.parse_json_folder(args.dets_dir, classes=[args.tracking_label])

        if len(dets) == 0:
            raise AssertionError("no annotation found. Maybe due to incorrect 'annotation_ext' parameter or no crop matching 'label' found.")
    else:
        from darknet import YoloDetector
        yolo = YoloDetector.from_dir(args.net_dir)
        dets = BoundingBoxes()

        for image in tqdm(list(args.image_dir.glob("*.jpg")), desc="Detection"):
            img = np.array(Image.open(image))
            img_h, img_w = img.shape[:2]
            predictions = yolo.predict(img, args.conf_threshold)
            dets += Parser.parse_yolo_darknet_detections(predictions, str(image.expanduser().resolve()), (img_w, img_h), classes=[args.tracking_label])
        
        if args.save_dets:
            dets.save(save_dir=args.dets_dir)

    if args.do_tracking:
        if args.compute_flow:
            flows = OpticalFlow.compute(args.image_list_path, mask_border=True, mask_egi=True)

            if args.save_flow:
                OpticalFlow.save(flows, args.optflow_path)
        else:
            flows = OpticalFlow.read(args.optflow_path)

        dets = track(dets, args.image_list_path, flows, args.conf_threshold, args.min_dets, args.dist_threshold)
    
    operose(dets, args.image_dir / "operose", args.label)