from pathlib import Path

from BoxLibrary import *
from my_library import *
from my_xml_toolbox import *
from argparse import ArgumentParser

from tqdm.contrib import tzip
import matplotlib.pyplot as plt



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

    args = parser.parse_args()

    args.image_dir = Path(args.image_dir).expanduser().resolve()
    args.optflow_path = args.image_dir / "optical_flow.txt"
    args.image_list_path = args.image_dir / "image_list.txt"
    args.dets_dir = args.image_dir / "predictions"

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

    return args


def associate(txt_file, flows, tracks: "list[Track]") -> "dict[str, list[Track]]":
    boxes = [t.barycenter_box() for t in tracks]
    images = read_image_txt_file(txt_file)
    (img_width, img_height) = image_size(images[0])
    out_tracks = {}

    for i, image in enumerate(images):
        (dx, dy) = OpticalFlow.traverse_backward(flows[:i+1], 0, 0)  # +1 !!!
        xmin = dx
        ymin = dy
        xmax = img_width + dx
        ymax = img_height + dy

        image_tracks = [t for j, t in enumerate(tracks) if boxes[j].centerIsIn((xmin, ymin, xmax, ymax))]
        image_tracks = [t.movedBy(-xmin, -ymin) for t in image_tracks]

        out_tracks[image] = image_tracks

    return out_tracks


def track(boxes, image_list, flows, conf_thresh, min_points, dist_thresh, crop_percent=3/100):
    images = read_image_txt_file(image_list)
    images = [str(Path(p).expanduser().resolve()) for p in images]
    tracker = Tracker(min_confidence=conf_thresh, min_points=min_points, dist_thresh=dist_thresh)

    # Python dict are ordered but safer to iterate over image_names as key than dict.items()
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for image_name, flow in tzip(images, flows, desc="Tracking", unit="image"):
        tracker.update(boxes_by_name[image_name], flow)

    dets = tracker.get_filtered_tracks()
    asso = associate(image_list, flows, dets)

    return dets, asso
    

if __name__ == "__main__":
    args = parse_args()

    if not args.image_list_path.is_file():
        images = sorted(list(args.image_dir.glob("*.jpg")))
        args.image_list_path.write_text("\n".join(str(p.expanduser().resolve()) for p in images))

    dets = Parser.parse_yolo_det_folder(args.dets_dir, args.image_dir, classes=[args.tracking_label])

    if len(dets) == 0:
        raise AssertionError("no annotation found. Maybe due to incorrect 'annotation_ext' parameter or no crop matching 'label' found.")
    
    flows = OpticalFlow.read(args.optflow_path)

    tracks, asso = track(dets, 
        args.image_list_path, flows, args.conf_threshold, args.min_dets, args.dist_threshold)

    X = []
    Y = []
    C = []

    out_flows = {}
    images = read_image_txt_file(args.image_list_path)
    for i, image_name in enumerate(images):
        out_flows[image_name] = OpticalFlow.traverse_backward(flows[:i+1], 0, 0)

    for t in tracks:
        for det in t.history:
            # print(det.getImageName())
            x, y, *_ = det.getAbsoluteBoundingBox(BBFormat.XYC)
            c = (x - out_flows[det.getImageName()][0]) / 1024

            X.append(x)
            Y.append(y)
            C.append(c)

    plt.figure()
    plt.scatter(X, Y, s=2, c=C, cmap="jet")
    plt.show()