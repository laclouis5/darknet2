try:
    import cv2 as cv
except:
    from skimage import io, filters, morphology
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import os
from threading import Thread

import argparse
import glob

from darknet import *
from my_library import *
from reg_plane import *
from BoxLibrary import *
from my_xml_toolbox import XMLTree
# from test import egi_mask, cv_egi_mask, create_dir

from pathlib import Path


def create_operose_result(args):
    (image, save_dir, network_params, plants_to_keep) = args

    config_file = network_params["cfg"]
    model_path = network_params["model"]
    meta_path = network_params["obj"]

    consort = "Bipbip"

    # Creates and populate XML tree, save plant masks as PGM and XLM file
    # for each images
    img_name = os.path.basename(image)

    try:
        image_egi = cv_egi_mask(cv.imread(image))
        im_in = Image.fromarray(image_egi)
    except:
        image_egi = egi_mask(io.imread(image))
        im_in = Image.fromarray(np.uint8(255 * image_egi))

    h, w = image_egi.shape[:2]

    # Perform detection using Darknet
    detections = performDetect(
        imagePath=image,
        configPath=config_file,
        weightPath=model_path,
        metaPath=meta_path,
        showImage=False)

    # XML tree init
    xml_tree = XMLTree(
        image_name=img_name,
        width=w,
        height=h)

    # For every detection save PGM mask and add field to the xml tree
    for detection in detections:
        name = detection[0]

        if (plants_to_keep is not None) and (name not in plants_to_keep):
            continue

        bbox = detection[2]
        box  = convertBack((bbox[0]), bbox[1], bbox[2], bbox[3])
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        xml_tree.add_mask(name)

        im_out = Image.new(mode="1", size=(w, h))
        region = im_in.crop(box)
        im_out.paste(region, box)

        image_name_out = "{}_{}_{}.png".format(
            consort,
            os.path.splitext(img_name)[0],
            str(xml_tree.plant_count-1))

        im_out.save(os.path.join(save_dir, image_name_out))

    xml_tree.save(save_dir)


def process_operose(image_path, network_params, save_dir="operose/", plants_to_keep=None, nb_proc=-1):
    create_dir(save_dir)

    def ArgsGenerator(image_path, network_params, save_dir, plants_to_keep):
        images = [os.path.join(image_path, item) for item in os.listdir(image_path) if os.path.splitext(item)[1] == ".jpg"]
        for image in images:
            yield (image, save_dir, network_params, plants_to_keep)

    args = ArgsGenerator(image_path, network_params, save_dir, plants_to_keep)

    Parallel(n_jobs=nb_proc, backend="multiprocessing")(delayed(create_operose_result)(arg) for arg in args)


def operose(txt_file, yolo, keep, thresh=0.25, save_dir="operose/", nproc=1):
    cache_dir = os.path.join(save_dir, "cache/")
    create_dir(save_dir)
    create_dir(cache_dir)
    create_dir(os.path.join(cache_dir, "detections/"))

    (cfg, weights, meta) = yolo.get_cfg_weight_meta()
    images = read_image_txt_file(txt_file)

    min_dets, max_dist = (5, 7.5/100)
    tracker = Tracker(thresh, min_dets, max_dist)

    (img_h, img_w) = cv.imread(images[0]).shape[:2]
    x_margin, y_margin = int(img_w * 5/100), int(img_h * 5/100)

    # Compute flows
    cached_flow = os.path.join(cache_dir, "optical_flow.csv")

    def compute_flow():
        if not os.path.isfile(cached_flow):
            OpticalFlowLK.generate(txt_file, cached_flow, mask_egi=True)

    if nproc != 1:
        optical_flow_thread = Thread(target=compute_flow)
        optical_flow_thread.start()
    else:
        compute_flow()

    # Compute detections and increment tracker
    boxes = performDetectOnTxtFile(txt_file, yolo, thresh, n_proc=nproc)
    # boxes = Parser.parse_yolo_det_folder(
    #     os.path.join(save_dir, "cache/detections"),
    #     os.path.dirname(txt_file))
    boxes.getBoundingBoxByClass(keep)
    boxes = BoundingBoxes([box for box in boxes
        if box.centerIsIn([x_margin, y_margin, img_w - x_margin, img_h - y_margin])])
    boxes.save(save_dir=os.path.join(save_dir, "cache/detections"))

    if nproc != 1:
        optical_flow_thread.join()

    optical_flows = OpticalFlow.read(cached_flow)

    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())
    for (index, image) in enumerate(images):
        image_boxes = boxes_by_name[image]
        tracker.update(image_boxes, optical_flows[index])

    boxes = tracker.get_filtered_boxes()
    boxes = box_association(boxes, images, optical_flows)
    boxes = BoundingBoxes([box for box in boxes
        if box.centerIsIn([x_margin, y_margin, img_w - x_margin, img_h - y_margin])])
    boxes.drawAllCenters("visus/")

    # Write stuff
    def inner(element):
        (image, image_boxes) = element

        image_name = os.path.basename(image)
        (img_h, img_w) = cv.imread(image).shape[:2]
        radius = int(5/100 * min(img_w, img_h) / 2)

        xml_tree = XMLTree(image_name, width=img_w, height=img_h)

        for box in image_boxes:
            label = box.getClassId()
            xml_tree.add_mask(label)

            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            rect = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]

            out_name = f"{os.path.splitext(image_name)[0]}_{xml_tree.plant_count-1}.png"

            stem_mask = Image.new(mode="1", size=(img_w, img_h))
            stem_mask.paste(Image.new(mode="1", size=(radius*2, radius*2), color=1), rect)
            stem_mask.save(os.path.join(save_dir, out_name))

            xml_tree.save(save_dir)

    Parallel(n_jobs=nproc, verbose=10)(
        delayed(inner)(element) for element in boxes.getBoxesBy(lambda box: box.getImageName()).items())


def operose_2(folder, keep=None, save_dir="operose/", n_jobs=-1):
    def inner(element):
        (image, image_boxes) = element

        image_name = Path(image).name
        (img_h, img_w) = cv.imread(image).shape[:2]
        radius = int(5/100 * min(img_w, img_h) / 2)

        xml_tree = XMLTree(image_name, width=img_w, height=img_h)

        for box in image_boxes:
            label = box.getClassId()
            xml_tree.add_mask(label)

            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            rect = [int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius]

            out_name = f"{Path(image).stem}_{xml_tree.plant_count-1}.png"

            stem_mask = Image.new(mode="1", size=(img_w, img_h))
            stem_mask.paste(Image.new(mode="1", size=(radius*2, radius*2), color=1), rect)
            stem_mask.save(os.path.join(save_dir, out_name))

            xml_tree.save(save_dir)

    create_dir(save_dir)
    boxes = Parser.parse_yolo_det_folder(folder, folder, classes=keep)
    boxes_by_name = boxes.getBoxesBy(lambda box: box.getImageName())

    Parallel(n_jobs, verbose=10)(
        delayed(inner)(element) for element in boxes_by_name.items())


def calibrate_folder(folder, save_dir="calibrated/", n_proc=-1):
    create_dir(save_dir)
    images = files_with_extension(folder, ".jpg")
    (img_h, img_w) = cv.imread(images[0]).shape[:2]
    (mapx, mapy) = basler3M_calibration_maps((img_w, img_h))

    def calibrate_image(image):
        save_name = os.path.join(save_dir, os.path.basename(image))
        img = cv.imread(image)
        calibrated = calibrate(img, mapx, mapy)
        cv.imwrite(save_name, calibrated)

    Parallel(n_jobs=n_proc, verbose=10)(delayed(calibrate_image)(image) for image in images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path where sequential images are stored.")
    parser.add_argument("yolo_dir", help="Path where cfg, weights and data are stored.")
    parser.add_argument("label", help="Label to keep.")

    parser.add_argument("--thresh", "-t", default=0.25,
        help="Confidence threshold for detection network.")
    parser.add_argument("--save_dir", "-s", default="operose/")
    parser.add_argument("--n_proc", "-j", default=1,
        help="Number of processes to launch. Default is sequential. If out of memory error consider decreasing the number of proc used.")

    args = parser.parse_args()

    yolo = YoloModelPath(args.yolo_dir)
    create_image_list_file(args.folder)
    image_file = os.path.join(args.folder, "image_list.txt")
    operose(image_file, yolo, args.label, args.thresh, args.save_dir, args.n_proc)


if __name__ == "__main__":
    folder = "/media/deepwater/DATA/Shared/Louis/datasets/test_set/"
    yolo = YoloModelPath("results/yolov4-tiny_10")
    # boxes = performDetectOnFolder(yolo, folder, conf_thresh=25/100)
    # boxes.save()
    operose_2(folder, save_dir="operose_test/")