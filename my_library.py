# Created by Louis LAC 2019

from collections import namedtuple, defaultdict
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from skimage import io, data, filters, feature, color, exposure, morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
try:
    import cv2 as cv
except:
    pass
import os
import sys
from PIL import Image
from BoxLibrary import *

from scipy.optimize import linear_sum_assignment
from scipy import stats
from collections.abc import MutableSequence

from reg_plane import fit_plane, reg_score, BivariateFunction, Equation
import image_transform as imtf

from tqdm.contrib import tenumerate
from tqdm import tqdm

from functools import reduce
from pathlib import Path


class Tracker:

    def __init__(self, min_confidence, min_points, dist_thresh):
        self.min_points = min_points
        self.min_confidence = min_confidence
        self.dist_thresh = dist_thresh
        self.tracks = []
        self.inactive_tracks = []
        # self.optical_flows = []
        self.life_time = 0
        self.max_inactive = 30
        self._ox = 0.0
        self._oy = 0.0

    def update(self, detections, optical_flow):
        self.life_time += 1
        # self.optical_flows.append(optical_flow)
        (nx, ny) = optical_flow(0.0, 0.0)
        self._ox += nx
        self._oy += ny

        tracked_boxes = self._get_all_boxes()
        moved_detections = BoundingBoxes([box.movedBy(self._ox, self._oy) for box in detections])
        # moved_detections = BoundingBoxes()
        # for det in detections:
            # (x, y, _, _) = det.getAbsoluteBoundingBox(BBFormat.XYC)
            # (mx, my) = OpticalFlow.traverse_backward(self.optical_flows, x, y)
            # moved_detections.append(det.movedBy(mx, my))
            # moved_detections.append(det.movedBy(self._ox, self._oy))

        matches, unmatched_dets, unmatched_tracks = self.coco_assignement(moved_detections, tracked_boxes)

        # Tracked that are matched, are active
        for (det_idx, trk_idx) in matches:
            self.tracks[trk_idx].append(moved_detections[det_idx])
            self.tracks[trk_idx].epochs_without_update = 0

        # New track, is active
        for det_idx in unmatched_dets:
            new_track = Track(history=[moved_detections[det_idx]])
            self.tracks.append(new_track)

        # Unmatched tracks, may be removed if inactive
        for trk_idx in unmatched_tracks:
            self.tracks[trk_idx].epochs_without_update += 1

        # Do something with inactive tracks
        for trk_idx in reversed(range(len(self.tracks))):
            if self.tracks[trk_idx].epochs_without_update > self.max_inactive:
                self.inactive_tracks.append(self.tracks.pop(trk_idx))

    def coco_assignement(self, detections, tracks):
        if len(detections) == 0:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=int), np.arange(len(tracks))
        if len(tracks) == 0:
            return np.empty((0, 0), dtype=int), np.arange(len(detections)), np.empty((0, 0), dtype=int)

        min_image_size = min(detections[0].getImageSize())
        max_dist = self.dist_thresh * min_image_size

        visited = [False] * len(tracks)
        matches = []
        ignored = set()

        detections = sorted(enumerate(detections),
            key=lambda element: element[1].getConfidence(),
            reverse=True)

        for (i, det) in detections:
            min_distance = sys.float_info.max
            j_min_distance = None

            for (j, track) in enumerate(tracks):
                distance = det.distance(track)

                if distance < min_distance:
                    min_distance = distance
                    j_min_distance = j

            if min_distance < max_dist:
                if not visited[j_min_distance]:
                    visited[j_min_distance] = True
                    matches.append([i, j_min_distance])
                else:
                    ignored.add(i)

        matches = np.empty((0, 2), dtype=int) if not matches else np.array(matches)
        unmatched_dets = [d for d in range(len(detections)) if d not in matches[:, 0] and d not in ignored]
        unmatched_tracks = [t for t in range(len(tracks)) if t not in matches[:, 1]]

        return matches, np.array(unmatched_dets), np.array(unmatched_tracks)

    def assignment_match_indices(self, detections, tracks):
        if len(tracks) == 0:
            return np.empty((0, 0), dtype=int), np.arange(len(detections)), np.empty((0, 0), dtype=int)
        if len(detections) == 0:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=int), np.arange(len(tracks))

        min_image_size = min(detections[0].getImageSize())
        max_dist = self.dist_thresh * min_image_size

        dist_matrix = np.array([[detection.distance(track) for track in tracks] for detection in detections])
        dist_matrix[dist_matrix >= max_dist] = np.finfo(float).max

        dets_indices, tracks_indices = linear_sum_assignment(dist_matrix)

        matches = np.array([[d, t] for (d, t) in zip(dets_indices, tracks_indices) if dist_matrix[d, t] < max_dist])

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)

        unmatched_dets = np.array([d for d in range(len(detections)) if d not in matches[:, 0]])
        unmatched_tracks = np.array([t for t in range(len(tracks)) if t not in matches[:, 1]])

        return matches, unmatched_dets, unmatched_tracks

    def _get_all_boxes(self):
        return BoundingBoxes([track.barycenter_box() for track in self.tracks])

    def get_filtered_boxes(self):
        return BoundingBoxes([track.barycenter_box() for track in (self.tracks + self.inactive_tracks)
            if len(track) > self.min_points
            and track.mean_confidence() > self.min_confidence
        ])

    def get_mahal_filtered_boxes(self):
        return BoundingBoxes([track.robust_barycenter() for track in (self.tracks + self.inactive_tracks)
            if len(track) > self.min_points
            and track.mean_confidence() > self.min_confidence
        ])

    def get_filtered_tracks(self):
        return [track for track in (self.tracks + self.inactive_tracks)
            if len(track) > self.min_points
            and track.mean_confidence() > self.min_confidence]

    def print_stats_for_tracks(self, tracks=None):
        if tracks is None:
            tracks = self.tracks + self.inactive_tracks

        for track in tracks:
            (x, y, _, _) = track.barycenter_box().getAbsoluteBoundingBox(format=BBFormat.XYC)
            print("Track {}: len: {}, pos: (x: {:.6}, y: {:.6}), conf: {:.6}".format(track.track_id, len(track), x, y, track.mean_confidence()))

    def __repr__(self):
        return "\n".join(f"{track}" for track in (self.tracks + self.inactive_tracks))


class Track(MutableSequence):
    track_id = 0

    def __init__(self, history=None):
        self.track_id = Track.track_id
        self.epochs_without_update = 0
        self.history = history or BoundingBoxes()

        Track.track_id += 1

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]

    def __setitem__(self, index, item):
        self.history[index] = item

    def __delitem__(self, key):
        del self.history[key]

    def insert(self, index, item):
        self.history.insert(index, item)

    def mean_confidence(self):
        nb_boxes = len(self.history)
        assert  nb_boxes > 0, "Track is empty, cannot compute mean confidence"
        return reduce(lambda acc, box: acc + box.getConfidence(), self.history, 0.0) / nb_boxes

    def barycenter_box(self):
        nb_boxes = len(self.history)
        assert nb_boxes > 0, "Track is empty, cannot compute barycenter"
        
        m_x, m_y, m_w, m_h = 0.0, 0.0, 0.0, 0.0
        ref_box = self.history[0]

        for box in self.history:
            (x, y, w, h) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            m_x += x
            m_y += y
            m_w += w
            m_h += h

        return BoundingBox(
            imageName="No name",
            classId=ref_box.getClassId(),
            x=m_x/nb_boxes, y=m_y/nb_boxes, w=m_w/nb_boxes, h=m_h/nb_boxes,
            typeCoordinates=CoordinatesType.Absolute,
            imgSize=ref_box.getImageSize(),
            bbType=ref_box.getBBType(),
            classConfidence=self.mean_confidence(),
            format=BBFormat.XYC)

    def robust_barycenter(self, to_keep=0.8):
        """
        Computes the robust barycenter after removing outliers in the sense of mahalanobis distance.
        """

        if len(self.history) < 3:
            return self.barycenter_box()

        to_keep = int(to_keep * len(self.history))
        boxes = np.array([box.getAbsoluteBoundingBox(BBFormat.XYC) for box in self.history])
        coordinates = boxes[:, :2]
        barycenter = coordinates.mean(axis=0)
        cov = np.cov(coordinates.transpose())
        inv_cov = np.linalg.inv(cov)
        squared_mahal_dists = np.matmul(
            np.matmul(coordinates - barycenter, inv_cov),
            (coordinates - barycenter).transpose()
        )
        squared_mahal_dists = np.diagonal(squared_mahal_dists)
        sorted_dist_indices = np.argsort(squared_mahal_dists)
        filtered_indices = sorted_dist_indices[:to_keep]
        filtered_boxes = boxes[filtered_indices]
        filtered_box = filtered_boxes.mean(axis=0)

        ref_box = self.history[0]
        return BoundingBox(
            imageName="No name",
            classId=ref_box.getClassId(),
            x=filtered_box[0], y=filtered_box[1], w=filtered_box[2], h=filtered_box[3],
            typeCoordinates=ref_box.getCoordinatesType(),
            imgSize=ref_box.getImageSize(),
            bbType=ref_box.getBBType(),
            classConfidence=self.mean_confidence(),
            format=BBFormat.XYC)

    def confidence_ellipse(self, n_std=2):
        coords = np.array([box.getAbsoluteBoundingBox(BBFormat.XYC) for box in self.history])
        (x, y) = coords[:, 0], coords[:, 1]
        return confidence_ellipse(x, y, n_std)

    def movedBy(self, dx, dy):
        return Track(history=[box.movedBy(dx, dy) for box in self.history])

    def __repr__(self):
        (x, y, _, _) = self.barycenter_box().getAbsoluteBoundingBox(BBFormat.XYC)
        return "Id: {}, len: {}, pos: (x: {:.6}, y: {:.6}), conf.: {:.4}".format(
            self.track_id, len(self), x, y, self.mean_confidence())


def associate_tracks_with_image(txt_file, optical_flow, tracker):
    images = read_image_txt_file(txt_file)
    optical_flows = OpticalFlow.read(optical_flow)
    (img_width, img_height) = image_size(images[0])
    output = defaultdict(list)

    tracks = tracker.get_filtered_tracks()
    bary_boxes = tracker.get_filtered_boxes()

    for (i, image) in enumerate(images):
        (dx, dy) = OpticalFlow.traverse_backward(optical_flows[:i+1], 0, 0)

        xmin = dx
        ymin = dy
        xmax = dx + img_width
        ymax = dy + img_height

        for track in tracks:
            if track.barycenter_box().centerIsIn((xmin, ymin, xmax, ymax)):
                track = track.movedBy(-xmin, -ymin)
                output[image].append(track)

    return output


def associate_boxes_with_image(txt_file, optical_flow, boxes):
    images = read_image_txt_file(txt_file)
    opt_flows = OpticalFlow.read(optical_flow)
    (img_width, img_height) = image_size(images[0])
    out_boxes = BoundingBoxes()

    for (i, image) in enumerate(images):
        (dx, dy) = OpticalFlow.traverse_backward(opt_flows[:i+1], 0, 0)  # +1 !!!
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


def gts_in_unique_ref(txt_file, folder, optical_flow, label):
    opt_flows = read_optical_flow(optical_flow)
    acc_flow = np.cumsum(opt_flows, axis=0)

    boxes = Parser.parse_yolo_gt_folder(folder, [label_to_number[label]])
    boxes.mapLabels(number_to_label)

    # boxes = Parser.parse_xml_folder(folder, ["mais_tige"])
    # boxes.mapLabels({"mais_tige": label})

    out_boxes = BoundingBoxes()

    images = []
    with open(txt_file, "r") as f:
        images = [c.strip() for c in f.readlines()]

    for i, image in enumerate(images[:1000]):
        (dx, dy) = acc_flow[i, :]
        label_boxes = boxes.getBoundingBoxesByImageName(image)

        out_boxes += label_boxes.movedBy(-dx, -dy)

    return out_boxes


def evaluate_aggr(detections, gts):
    """
    detections: detections for all successive images
    gts: ground truths for selected images
    """

    # Filter detections
    image_names = gts.getNames()
    detections = BoundingBoxes([det for det in detections if det.getImageName() in gts.getNames()])
    Evaluator().printAPsByClass((detections + gts), thresh=7.5/100, method=EvaluationMethod.Distance)


def confidence_ellipse(x, y, n_std=1):
    """
    Returns the confidence ellipse for 2 correlated distributions up
    to some confidence n * sigma.

    Params:
    - x (1D array): first distribution
    - y (1D array): secon distribution
    - n_std (int): confidence as `n * sigma`

    Returns:
    - (x, y): center of the ellipse
    - (w, h): width and height of the ellipse
    - angle: inclinaison of the ellispe
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    covariance = np.cov(x, y)
    eig_vals, eig_vects = eigsorted(covariance)
    eig_vals = np.maximum(0, eig_vals)

    (ex, ey) = np.mean(x), np.mean(y)
    # (w, h) = 2 * n_std * np.sqrt(eig_vals)
    q = 2 * stats.norm.cdf(n_std) - 1
    r2 = stats.chi2.ppf(q, 2)
    (w, h) = 2 * np.sqrt(eig_vals * r2)

    # r = stats.f.ppf(0.95, 2, len(x) - 2) * 2 / (len(x) - 2)  # Rajouter facteur !!!
    # (w, h) = np.sqrt(eig_vals * r)  # demi-grand axe

    angle = np.degrees(np.arctan2(*eig_vects[:, 0][::-1]))

    return (ex, ey, w, h, angle)

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def box_association(boxes, images, opt_flows):
    (img_width, img_height) = image_size(images[0])
    out_boxes = BoundingBoxes()

    for i, image in enumerate(images):
        (dx, dy) = OpticalFlow.traverse_backward(opt_flows[:i+1], 0, 0)  # +1 !!!
        xmin = dx
        ymin = dy
        xmax = img_width + dx
        ymax = img_height + dy

        image_boxes = boxes.boxes_in([xmin, ymin, xmax, ymax])
        image_boxes = image_boxes.movedBy(-xmin, -ymin)

        for box in image_boxes:
            (x, y, w, h) = box.getAbsoluteBoundingBox()
            out_boxes.append(BoundingBox(imageName=image, classId=box.getClassId(), x=x, y=y, w=w, h=h, imgSize=box.getImageSize(), bbType=BBType.Detected, classConfidence=box.getConfidence()))

    return out_boxes


def move_gts_in_unique_ref(boxes, txt_file, opt_flow_file):
    images = read_image_txt_file(txt_file)
    flows = OpticalFlow.read(opt_flow_file)
    flows = [Function(
        fn1=lambda x, y, fl=flow[0]: fl,
        fn2=lambda x, y, fl=flow[1]: fl) for flow in flows]

    def deplacement_for_coord(x, y, flow):
        x0, y0 = x, y
        for f in reversed(flow):
            (vx, vy) = f(x0, y0)
            x0 += vx
            y0 += vy
        return (x0 - x, y0 - y)

    for (index, image) in enumerate(images):
        image_boxes = boxes.getBoundingBoxesByImageName(image)
        for box in image_boxes:
            (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)
            (mx, my) = deplacement_for_coord(x, y, flows[:index+1])
            box.moveBy(mx, my)
            box._imageName = "No name"

    return boxes


def optical_flow_visualisation(txt_file):
    images = []
    with open(txt_file, "r") as f:
        images = [c.strip() for c in f.readlines()]

    image_1 = cv.imread(images[10])
    image_2 = cv.imread(images[11])
    # image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2RGB)
    # image_2 = cv.cvtColor(image_2, cv.COLOR_BGR2RGB)
    # image_1 = cv.resize(image_1, (632, 632), interpolation=cv.INTER_AREA)
    # image_2 = cv.resize(image_2, (632, 632), interpolation=cv.INTER_AREA)

    egi_2 = egi_mask(image_2)

    opt_flow, _ = optical_flow(image_1, image_2)
    x_flow = opt_flow[:, :, 0]
    y_flow = opt_flow[:, :, 1]

    percent = 0.2
    (img_h, img_w) = image_2.shape[:2]
    h_start = int(img_h * percent)
    h_stop = int(img_h * (1 - percent))
    w_start = int(img_w * percent)
    w_stop = int(img_w * (1 - percent))

    # x_flow = np.array(x_flow[h_start:h_stop, w_start+70:w_stop+70])
    # data = x_flow[~egi_2[h_start:h_stop, w_start+70:w_stop+70]]
    # data = x_flow * ~egi_2[h_start:h_stop, w_start+70:w_stop+70]

    # abs = np.arange(632)[w_start+70:w_stop+70]

    plt.figure()
    # plt.hist(x_flow.ravel(), 200)
    plt.imshow(x_flow)
    # plt.plot(np.array(x_flow[800:1000, :1400]).mean(axis=0))
    # plt.ylim([0, 30])
    # plt.imshow(data)
    plt.show()


def convert_to_grayscale(image):
    '''
    Convert an image (numpy array) to grayscale.
    '''
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def optical_flow(image_1, image_2, prev_opt_flow=None):
    first_image = convert_to_grayscale(image_1)
    second_image = convert_to_grayscale(image_2)
    (img_h, img_w) = image_1.shape[:2]

    flag = cv.OPTFLOW_USE_INITIAL_FLOW

    if prev_opt_flow is None:
        prev_opt_flow = np.zeros((img_h, img_w, 2), dtype=np.float32)
        flag = 0

    optical_flow = cv.calcOpticalFlowFarneback(
        prev=first_image,
        next=second_image,
        flow=prev_opt_flow,
        pyr_scale=0.5,
        levels=4,
        winsize=8,
        iterations=4,
        poly_n=5,
        poly_sigma=1.1,
        flags=flag)

    return optical_flow, image_2


def opt_flow_plane(txt_file):
    images = []
    with open(txt_file, "r") as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    first_image = cv.imread(images[0])
    (img_h, img_w) = first_image.shape[:2]

    xmin = int(img_w * 0.2)
    xmax = int(img_w * 0.9)
    ymin = int(img_h * 0.1)
    ymax = int(img_h * 0.9)

    base_mask = np.full(first_image.shape[:2], False)
    base_mask[ymin:ymax:, xmin:xmax] = True

    opflow = None

    X, Y = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))

    opt_flows = "0 0\n"

    for image in images[1:]:
        second_image = cv.imread(image)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        egi = np.uint8(egi_mask(first_image) * 255)
        egi = cv.dilate(egi, kernel, iterations=5)
        egi = egi.astype(np.bool)

        mask = ~egi & base_mask

        opflow, tmp = optical_flow(first_image, second_image,
            prev_opt_flow=opflow)

        dX, dY = opflow[..., 0], opflow[..., 1]

        # should be updated with Equation
        coeffs_X = fit_plane(X, Y, dX, mask).coeffs
        coeffs_Y = fit_plane(X, Y, dY, mask).coeffs

        c_x = coeffs_X[0] * (img_w / 2) + coeffs_X[1] * (img_h / 2) + coeffs_X[2]
        c_y = coeffs_Y[0] * (img_w / 2) + coeffs_Y[1] * (img_h / 2) + coeffs_Y[2]

        opt_flows += "{} {}\n".format(c_x, c_y)
        # fX = coeffs_X[0] * X + coeffs_X[1] * Y + coeffs_X[2]
        # fY = coeffs_Y[0] * X + coeffs_Y[1] * Y + coeffs_Y[2]
        #
        # R2_X = reg_score(dX, fX, mask)
        # R2_Y = reg_score(dY, fY, mask)
        #
        # c_x = coeffs_X[0] * img_w / 2 + coeffs_X[1] * img_h / 2 + coeffs_X[2]
        # c_y = coeffs_Y[0] * img_w / 2 + coeffs_Y[1] * img_h / 2 + coeffs_Y[2]
        #
        # print("Eq. X: {:.3}*X + {:.3}*Y + {:.3}".format(*coeffs_X))
        # print("Eq. Y: {:.3}*X + {:.3}*Y + {:.3}".format(*coeffs_Y))
        # print("R² X: {:.2%}".format(R2_X))
        # print("R² Y: {:.2%}".format(R2_Y))
        # print("Optical Flow Center X: {}".format(c_x))
        # print("Optical Flow Center Y: {}".format(c_y))
        # print()
        #
        # from mpl_toolkits import mplot3d
        # x = X[::8, ::8].reshape(-1, 1)
        # y = Y[::8, ::8].reshape(-1, 1)
        # z1 = (dX[::8, ::8] * mask[::8, ::8]).reshape(-1, 1)
        # z2 = (dY[::8, ::8] * mask[::8, ::8]).reshape(-1, 1)
        # colors = cv.cvtColor(first_image[::8, ::8, :], cv.COLOR_BGR2RGB)
        # colors  = colors.reshape(-1, 3) / 255
        #
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # ax.scatter(x, y, z1,
        #     s=2,
        #     facecolors=colors)
        # ax.plot_surface(X, Y, fX, color=[0, 0, 1, 0.3])
        # plt.show()

        first_image = tmp

    with open("planar_opt_flow.txt", "w") as f:
        f.write(opt_flows)


def get_border_mask(image_size):
    mask = np.full(image_size, True)

    (h, w) = image_size
    xmin = int(w * 0.2)
    xmax = int(w * 0.9)
    ymin = int(h * 0.1)
    ymax = int(h * 0.9)

    mask[ymin:ymax:, xmin:xmax] = False
    return ~mask


def image_displacement(img1, img2, prev_opt_flow=None, masking_border=False, mask_egi=False):
    mask = np.full(img1.shape[:2], True)

    if masking_border:
        mask = get_border_mask(img1.shape[:2])

    if mask_egi:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

        egi = np.uint8(egi_mask(img1) * 255)
        egi = cv.dilate(egi, kernel, iterations=5)
        egi = egi.astype(np.bool)

        mask = mask & ~egi

    flow, _ = optical_flow(img1, img2, prev_opt_flow)

    return median_opt_flow(flow, mask), flow


class OpticalFlow:

    def __init__(self, mask_border=False, mask_egi=False):
        self.mask_egi = mask_egi
        self.mask_border = mask_border
        self.previous_flow = None
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        self.fl = cv.cuda_FarnebackOpticalFlow.create(
            numLevels=4,
            pyrScale=0.5,
            winSize=16,
            numIters=4,
            polyN=5,
            polySigma=1.1,
            flags=cv.OPTFLOW_USE_INITIAL_FLOW)

    def optical_flow(self, img1, img2):
        img_h, img_w = img1.shape[:2]

        first_frame = cv.cuda_GpuMat(img1)
        second_frame = cv.cuda_GpuMat(img2)

        first_frame = cv.cuda.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        second_frame = cv.cuda.cvtColor(second_frame, cv.COLOR_BGR2GRAY)

        if self.previous_flow is None:
            self.previous_flow = cv.cuda_GpuMat(img_h, img_w, cv.CV_32FC2)

        flow = self.fl.calc(first_frame, second_frame, self.previous_flow)
        self.previous_flow = flow

        return flow.download()

    def displacement(self, img1, img2):
        flow = self.optical_flow(img1, img2)
        mask = self.mask(img1, img2)
        return self.median(flow, mask)

    def displacement_eq(self, img1, img2):
        (x, y) = self.displacement(img1, img2)
        return BivariateFunction(lambda x, y, ex=x, ey=y: (ex, ey))

    def displacement_plane_equation(self, img1, img2):
        image_size = img1.shape[:2]
        flow = self.optical_flow(img1, img2)

        dx = flow[..., 0]
        dy = flow[..., 1]

        (x_grid, y_grid) = np.meshgrid(
            np.arange(0, image_size[1]),
            np.arange(0, image_size[0])
        )

        mask = self.mask(img1, img2)

        eq_x = fit_plane(x_grid, y_grid, dx, mask)
        eq_y = fit_plane(x_grid, y_grid, dy, mask)

        return eq_x, eq_y

    def mask(self, img1, img2):
        if not self.mask_egi:
            return self.get_border_mask(img1.shape[:2])

        egi1 = self.egi_mask(img1)
        egi2 = self.egi_mask(img2)
        egi = egi1 & egi2

        if self.mask_border:
            return self.get_border_mask(img1.shape[:2]) & egi
        else:
            return egi

    def egi_mask(self, img):
        egi = np.uint8(egi_mask(img) * 255)
        egi = cv.dilate(egi, self.kernel, iterations=2)
        egi = egi.astype(np.bool)
        return ~egi

    @classmethod
    def get_border_mask(cls, image_size):
        mask = np.full(image_size, True)

        (h, w) = image_size
        xmin = int(w * 0.2)
        xmax = int(w * 0.9)
        ymin = int(h * 0.1)
        ymax = int(h * 0.9)

        mask[ymin:ymax:, xmin:xmax] = False

        return ~mask

    @classmethod
    def mean(cls, optical_flow, mask=None):
        """
        `mask` is a binary mask where locations where to compute optical_flow
        ar marked as True.
        """
        dx = optical_flow[..., 0]
        dy = optical_flow[..., 1]

        if mask is not None:
            dx = dx[mask == True]
            dy = dy[mask == True]

        return np.mean(dx), np.mean(dy)

    @classmethod
    def median(cls, optical_flow, mask=None):
        """
        `mask` is a binary mask where locations where to compute optical_flow
        ar marked as True.
        """
        dx = optical_flow[..., 0]
        dy = optical_flow[..., 1]

        if mask is not None:
            dx = dx[mask == True]
            dy = dy[mask == True]

        return np.median(dx), np.median(dy)

    @staticmethod
    def compute(image_list, mask_border=False, mask_egi=False):
        images = read_image_txt_file(image_list)
        past_image = cv.imread(images[0])
        data = [BivariateFunction(lambda x, y: (0.0, 0.0))]
        ofCalc = OpticalFlow(mask_border=mask_border, mask_egi=mask_egi)

        for image in tqdm(images[1:], desc="Optical Flow", unit="image"):
            current_image = cv.imread(image)
            dx, dy = ofCalc.displacement(current_image, past_image)
            data.append(BivariateFunction(lambda x, y, ex=dx, ey=dy: (ex, ey)))
            past_image = current_image

        return data

    @staticmethod
    def save(flows, path):
        flows = (t(0.0, 0.0) for t in flows)
        flows = (f"{x}, {y}" for x, y in flows)
        Path(path).write_text("\n".join(flows))        

    @staticmethod
    def generate(txt_file, name="opt_flow.txt", save_dir=None, masking_border=False, mask_egi=False):
        images = read_image_txt_file(txt_file)
        past_image = cv.imread(images[0])
        data = "0.000000, 0.000000\n"
        ofCalc = OpticalFlow(mask_border=masking_border, mask_egi=mask_egi)

        for image in tqdm(images[1:], desc="Optical Flow", unit="image"):
            current_image = cv.imread(image)
            dx, dy = ofCalc.displacement(current_image, past_image)
            line = f"{dx}, {dy}\n"
            # print("(dx: {:.4}, dy: {:.4})".format(dx, dy))
            data += line
            past_image = current_image

        save_dir = save_dir or "save/"
        create_dir(save_dir)

        with open(os.path.join(save_dir, name), "w") as f:
            f.write(data)

    @classmethod
    def generate_plane(cls, txt_file, name="opt_flow_plane.txt", mask_border=False, mask_egi=False):
        images = read_image_txt_file(txt_file)
        past_image = cv.imread(images[0])
        of_calc = OpticalFlow(mask_border=mask_border, mask_egi=mask_egi)
        data = "0, 0, 0, 0, 0, 0\n"

        for image in images[1:]:
            current_image = cv.imread(image)
            (eq_x, eq_y) = of_calc.displacement_plane_equation(current_image, past_image)
            (xx, xy, x0) = eq_x.coeffs
            (yx, yy, y0) = eq_y.coeffs
            line = f"{xx}, {xy}, {x0}, {yx}, {yy}, {y0}\n"
            # print(line)
            data += line
            past_image = current_image

        with open(name, "w") as f:
            f.write(data)

    @classmethod
    def read(cls, file):
        flows = []
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(",")
                dx, dy = float(line[0]), float(line[1])
                flows.append(BivariateFunction(
                    lambda x, y, ex=dx, ey=dy: (ex, ey)
                ))
        return flows

    @classmethod
    def read_planes(cls, file):
        flows = []
        with open(file, "r") as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split(",")
                coeffs = [float(coeff) for coeff in line]
                eq_x = Equation(coeffs[:3])
                eq_y = Equation(coeffs[3:])
                flows.append(BivariateFunction(lambda x, y, ex=eq_x, ey=eq_y: (ex(x, y), ey(x, y))))
        return flows

    @staticmethod
    def traverse_backward(flows, x, y):
        x0, y0 = x, y
        for f in reversed(flows):
            (vx, vy) = f(x0, y0)
            x0 += vx
            y0 += vy
        return (x0 - x, y0 - y)


class OpticalFlowLK:
    feature_params = {
        "maxCorners": 1000,
        "qualityLevel": 0.01,
        "minDistance": 16,
        "blockSize": 32,
    }

    lk_params = {
        "winSize": (16, 16),
        "maxLevel": 5,
    }

    def __init__(self):
        self.prev_dx = 0.0
        self.prev_dy = 0.0

    def __call__(self, prev, next, mask_egi=False, mask_border=False):
        mask = None
        
        if mask_egi:
            mask = 255 - cv_egi_mask(next)

        if mask_border:
            (img_h, img_w) = prev.shape[:2]
            border_mask = np.full((img_h, img_w), 0, dtype=np.uint8)
            m_x, m_y = int(0.1 * img_w), int(0.1 * img_h)
            border_mask[m_y:img_h-m_y, 2*m_x:img_w-m_x] = 255
            mask = mask & border_mask if egi_mask else border_mask

        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

        p0 = cv.goodFeaturesToTrack(prev_gray,
            mask=mask, **self.feature_params).astype(np.float32)
        p1_est = (p0 + [self.prev_dx, self.prev_dy]).astype(np.float32)

        # for p in p1_est.reshape(-1, 2):
        #     cv.circle(next, center=(p[0], p[1]), radius=2, color=(255, 0, 0), thickness=-1)

        p1, _, _ = cv.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, p1_est,
            cv.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        p0r, _, _ = cv.calcOpticalFlowPyrLK(next_gray, prev_gray, p1,
            None, **self.lk_params)

        p0 = p0.reshape(-1, 2)
        p0r = p0r.reshape(-1, 2)
        p1 = p1.reshape(-1, 2)

        keep = (abs(p0r - p0) < 2).max(1)
        p0_ok = p0[keep == 1]
        p1_ok = p1[keep == 1]
        v = (p1_ok - p0_ok)
        (m_x, m_y) = np.median(v, axis=0)
        ratio = 2.5
        keep =  (v[..., 0] > m_x - ratio) & \
            (v[..., 0] < m_x + ratio) & \
            (v[..., 1] > m_y - ratio) & \
            (v[..., 1] < m_y + ratio)

        # for p in p0:
        #     cv.circle(next, center=(p[0], p[1]), radius=2, color=(0, 0, 255), thickness=-1)
        # for p in p1_ok[keep == 1]:
        #     cv.circle(next, center=(p[0], p[1]), radius=2, color=(0, 255, 0), thickness=-1)
        # for (p_1, p_2) in zip(p0_ok[keep == 1], p1_ok[keep == 1]):
        #     cv.line(next,
        #         (p_1[0], p_1[1]), (p_2[0], p_2[1]), color=(0, 255, 0), thickness=1)

        # cv.imshow("image", next)
        # key = cv.waitKey(1) & 0xFF
        # if key == ord("q"): return

        try:
            t, _ = cv.estimateAffine2D(p0_ok[keep == 1], p1_ok[keep == 1])
            self.prev_dx, self.prev_dy = t[0][2], t[1][2]
        except:
            print("Not enouth points to compute flow. Using old value.")

        # print(self.prev_dx, self.prev_dy)

        return self.prev_dx, self.prev_dy

    @staticmethod
    def generate(txt_file, file_name="optical_flow.csv", mask_egi=False, mask_border=False):
        images = read_image_txt_file(txt_file)
        prev = cv.imread(images[0])
        opt_flow = OpticalFlowLK()
        flows = [(0.0, 0.0)]

        for image in images[1:]:
            next = cv.imread(image)
            dx, dy = opt_flow(next, prev, mask_egi, mask_border)
            flows.append((dx, dy))

            prev = next

        with open(file_name, "w") as f:
            for (dx, dy) in flows:
                f.write(f"{dx},{dy}\n")

    @staticmethod
    def read(csv_file):
        flows = []

        with open(csv_file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(",")
                dx, dy = float(line[0]), float(line[1])
                flows.append(BivariateFunction(
                    lambda x, y, ex=dx, ey=dy: (ex, ey)
                ))
        return flows


def egi_mask(image, thresh=35):
    '''
    Takes as input a numpy array describing an image and return a
    binary mask thresholded over the Excess Green Index.
    '''
    img_h, img_w = image.shape[:2]
    small_area = int(0.1 / 100 * img_w * img_h)

    image_np  = np.array(image).astype(np.float)
    image_egi = np.sum(np.array([-1, 2, -1]) * image_np, axis=2)
    image_gf  = filters.gaussian(image_egi, sigma=1, mode="reflect")
    image_out = image_gf > thresh
    image_out = morphology.remove_small_objects(image_out, small_area)
    image_out = morphology.remove_small_holes(image_out, small_area)
    return image_out

def cv_egi_mask(image, thresh=40):
    '''
    Takes as input a numpy array describing an image and return a
    binary mask thresholded over the Excess Green Index. OpenCV implementation.
    '''
    image_np = np.array(image).astype(np.float32)
    image_np = 2 * image_np[:, :, 1] - image_np[:, :, 0] - image_np[:, :, 2]

    image_bin = image_np > thresh

    nb_components, output, stats, _ = cv.connectedComponentsWithStats(image_bin.astype(np.uint8), connectivity=8)

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img_out = np.zeros_like(output, dtype=np.uint8)

    for i in range(nb_components):
        if sizes[i] >= 500:
            img_out[output == i + 1] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    image_morph = cv.morphologyEx(img_out, op=cv.MORPH_CLOSE, kernel=kernel)

    return image_morph.astype(np.uint8)


def create_dir(directory):
    '''
    Creates the spedified directory if doesn't exist.
    '''
    if not os.path.isdir(directory):
        os.mkdir(directory)


# Obsolete
def yolo_det_to_bboxes(image_name, yolo_detections):
    '''
    Takes as input a list of tuples (label, conf, x, w, w, h) predicted by
    yolo framework and returns a boundingBoxes object representing the boxes.
    image_name is required.
    '''
    bboxes = []

    for detection in yolo_detections:
        label = detection[0]
        confidence = detection[1]
        box = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])

        bbox = BoundingBox(imageName=image_name, classId=label, x=xmin, y=ymin, w=xmax, h=ymax, typeCoordinates=CoordinatesType.Absolute, classConfidence=confidence, bbType=BBType.Detected, format=BBFormat.XYX2Y2)

        bboxes.append(bbox)

    return BoundingBoxes(bounding_boxes=bboxes)

# Obsolete: see BoundingBoxes.save() and BoundingBox.description()
def save_bboxes_to_txt(bounding_boxes, save_dir):
    '''
    Saves boxes wrapped in a boundingBoxes object to a yolo annotation file in
    the specified save_dir directory. Format is XYX2Y2 abs (hard coded).
    '''
    # Saves all detections in a BBoxes object as txt file
    names = bounding_boxes.getNames()

    for name in names:
        boxes = bounding_boxes.getBoundingBoxesByImageName(name)
        boxes = [box for box in boxes if box.getBBType() == BBType.Detected]

        string = ""
        for box in boxes:
            label = box.getClassId()
            conf = box.getConfidence()
            (xmin, ymin, xmax, ymax) = box.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

            string += "{} {} {} {} {} {}\n".format(label, str(conf), str(xmin), str(ymin), str(xmax), str(ymax))

        save_name = os.path.splitext(boxes[0].getImageName())[0] + ".txt"

        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(string)

# Obsolete: see Parser class
def read_detection_txt_file(file_path, img_size=None):
    '''
    Takes a detection file and its correponding image size and returns
    a boundingBoxes object representing boxes. Detection file is XYX2Y2 abs
    (hard coded).
    '''
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, conf, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Absolute, format=BBFormat.XYX2Y2, imgSize=img_size, bbType=BBType.Detected, classConfidence=conf))

    return bounding_boxes

# Obsolete: see Parser class
def read_gt_annotation_file(file_path, img_size=None):
    '''
    Takes a yolo GT file and its correponding image size and returns
    a boundingBoxes object representing boxes. Yolo format is XYWH relative,
    image_size must be provided.
    '''
    bounding_boxes = BoundingBoxes(bounding_boxes=[])
    image_name = os.path.basename(os.path.splitext(file_path)[0] + '.jpg')

    with open(file_path, 'r') as f:
        content = f.readlines()
        content = [line.strip().split() for line in content]

    for det in content:
        (label, x, y, w, h) = det[0], float(det[1]), float(det[2]), float(det[3]), float(det[4])
        bounding_boxes.addBoundingBox(BoundingBox(imageName=image_name, classId=label, x=x, y=y, w=w, h=h, typeCoordinates=CoordinatesType.Relative, imgSize=img_size))

    return bounding_boxes

# Obsolete: see Parser class
def parse_yolo_folder(data_dir):
    '''
    Parsed a folder containing yolo GT annotations and their corresponding
    images with the same name. Returns a boundingBoxes object.
    '''
    annotations = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.splitext(item)[1] == '.txt']
    images = [os.path.splitext(item)[0] + '.jpg' for item in annotations]
    bounding_boxes = BoundingBoxes(bounding_boxes=[])

    for (img, annot) in zip(images, annotations):
        img_size = Image.open(img).size
        image_boxes = read_gt_annotation_file(annot, img_size)
        [bounding_boxes.addBoundingBox(bb) for bb in image_boxes.getBoundingBoxes()]

    return bounding_boxes


def xywh_to_xyx2y2(x, y, w, h):
    '''
    Takes as input absolute coords and returns integers.
    '''
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def xywh_to_xyx2y2_float(x, y, w, h):
    '''
    Takes as input absolute coords and returns integers.
    '''
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax


def xyx2y2_to_xywh(xmin, ymin, xmax, ymax):
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h

# Obsolete
def save_yolo_detect_to_txt(yolo_detections, save_name):
    """
    Takes a list of yolo detections (tuples returned by the framework) and
    saved those detections in XYX2Y2 abs format in save_name file.
    """
    lines = []

    for detection in yolo_detections:
        box = detection[2]
        (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(box[0], box[1], box[2], box[3])
        confidence = detection[1]
        lines.append("{} {} {} {} {} {}\n".format(detection[0], confidence, xmin, ymin, xmax, ymax))

    with open(save_name, 'w') as f:
        f.writelines(lines)


def nms(bboxes, conf_thresh=0.25, nms_thresh=0.3):
    """
    Wrapper for OpenCV NMS algorithm.

    Parameters:
        bboxes (BoundingBox):
            The boxes for one particular image containing duplicate boxes.
        conf_thresh (optional float):
            The threshold on detected box confidence to remove false detections.
        nms_thresh (optional float):
            The threshold on box Intersection over Union for box merging. A value near 1 is less permissive than a value close to 0 as the IoU has to be higher to merge boxes.

    Returns:
        BoundingBoxes: The filtered boxes.
    """
    assert len(bboxes.getNames()) <= 1, "Func nms should be used on BoundingBoxes representing only one image. Image names received: {}".format(bboxes.getNames())

    labels = bboxes.getClasses()
    filtered_boxes = BoundingBoxes()

    for label in labels:
        boxes_label = bboxes.getBoundingBoxByClass(label)
        boxes = [box.getAbsoluteBoundingBox(BBFormat.XYWH) for box in boxes_label]
        boxes = [[box[0], box[1], box[2], box[3]] for box in boxes]
        conf  = [box.getConfidence() for box in boxes_label]

        indices = cv.dnn.NMSBoxes(boxes, conf, conf_thresh, nms_thresh)
        indices = [index for list in indices for index in list]

        boxes_to_keep = np.array(boxes_label)[indices]
        filtered_boxes += boxes_to_keep.tolist()

    return filtered_boxes


def remap_yolo_GT_file_labels(file_path, to_keep):
    """
    Takes path to yolo GT file, reads the file and removes lines with
    the labels specified in the to_keep mapping list.
    """
    content = []
    with open(file_path, "r") as f_read:
        content = f_read.readlines()

    content = [c.strip().split() for c in content]
    content = [line for line in content if (line[0] in to_keep.keys())]

    with open(file_path, "w") as f_write:
        for line in content:
            f_write.write("{} {} {} {} {}\n".format(to_keep[line[0]], line[1], line[2], line[3], line[4]))


def remap_yolo_GT_files_labels(folder, to_keep):
    files = [os.path.join(folder, item) for item in os.listdir(folder) if os.path.splitext(item)[1] == ".txt"]
    for file in files:
        remap_yolo_GT_file_labels(file, to_keep)


def crop_annotation_to_square(annot_folder, save_dir, lab_to_name):
    annotations = [os.path.join(annot_folder, item) for item in os.listdir(annot_folder) if os.path.splitext(item)[1] == '.txt']

    for annotation in annotations:
        content_out = []
        corresp_img = os.path.splitext(annotation)[0] + '.jpg'
        (img_w, img_h) = Image.open(corresp_img).size

        print("In landscape mode: {} by {}".format(img_w, img_h))
        # Here are abs coords of square bounds (left and right)
        (w_lim_1, w_lim_2) = round(float(img_w)/2 - float(img_h)/2), round(float(img_w)/2 + float(img_h)/2)

        with open(annotation, 'r') as f:
            print("Reading annotation...")
            content = f.readlines()
            content = [line.strip() for line in content]

            for line in content:
                print("Reading a line...")
                line = line.split()
                # Get relative coords (in old coords system)
                (label, x, y, w, h) = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                print("Line is: {} {} {} {} {}".format(label, x, y, w, h))

                # If bbox is not out of the new square frame
                if not (x*img_w < w_lim_1 or x*img_w > w_lim_2):
                    print("In square bounds")
                    # But if bbox spans out of one bound (l or r)
                    if (x - w / 2.0) < float(w_lim_1) / img_w:
                        print("Spans out of left bound")
                        # Then adjust bbox to fit in the square
                        w = w - (float(w_lim_1) / img_w - (x - w / 2.0))
                        x = float(w_lim_1 + 1) / img_w + w / 2.0
                    if (x + w / 2.0) > float(w_lim_2) / img_w:
                        print("Span out of right bound")
                        w = w - (x + w / 2.0 - float(w_lim_2) / img_w)
                        x = float(w_lim_2) / img_w - w / 2.0
                    else:
                        print("Does not spans outside")

                # If out of bounds...
                else:
                    print("Out of square bounds")
                    # ...do not process the line
                    continue

                # Do not forget to convert from old coord sys to new one
                x = (x * img_w - float(w_lim_1)) / float(w_lim_2 - w_lim_1)
                w = w * img_w / float(w_lim_2 - w_lim_1)

                assert x >= 0, "Value was {}".format(x)
                assert x <= 1, "Value was {}".format(x)
                assert (x - w / 2) >= 0, "Value was {}".format(x - w / 2)
                assert (x + w / 2) <= 1, "Value was {}".format(x + w / 2)

                size = min(img_w, img_h)

                (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(x * size, y * size, w * size, h * size)

                new_line = "{} {} {} {} {}\n".format(lab_to_name[label], xmin, ymin, xmax, ymax)
                content_out.append(new_line)

        # Write updated content to TXt file
        with open(os.path.join(save_dir, os.path.basename(annotation)), 'w') as f:
            f.writelines(content_out)


def crop_detection_to_square(image_path, save_dir, model, config_file, meta_file):
    images = files_with_extension(image_path, ".jpg")
    images.sort()

    for image in images:
        content_out = []
        (img_w, img_h) = Image.open(image).size
        (w_lim_1, w_lim_2) = round(float(img_w) / 2 - float(img_h) / 2), round(float(img_w) / 2 + float(img_h) / 2)

        detections = performDetect(
            imagePath=image,
            configPath=config_file,
            weightPath=model,
            metaPath=meta_file,
            showImage=False)

        for detection in detections:
            label = detection[0]
            prob = detection[1]
            (x, y, w, h) = detection[2]
            (x, y, w, h) = (x / img_w, y / img_h, w / img_w, h / img_h)

            # If bbox is not out of the new square frame
            if (x * img_w < w_lim_1 or x * img_w > w_lim_2): continue

            # But if bbox spans out of one bound (l or r)
            if x - w / 2.0 < float(w_lim_1) / img_w:
                # Then adjust bbox to fit in the square
                w = w - (float(w_lim_1) / img_w - (x - w / 2.0))
                x = float(w_lim_1 + 1) / img_w + w / 2.0
            if x + w / 2.0 > float(w_lim_2) / img_w:
                w = w - (x + w / 2.0 - float(w_lim_2) / img_w)
                x = float(w_lim_2) / img_w - w / 2.0

            # Do not forget to convert from old coord sys to new one
            x = (x * img_w - float(w_lim_1)) / float(w_lim_2 - w_lim_1)
            w = w * img_w / float(w_lim_2 - w_lim_1)

            assert x >= 0, "Value was {}".format(x)
            assert x <= 1, "Value was {}".format(x)
            assert (x - w / 2) >= 0, "Value was {}".format(x - w / 2)
            assert (x + w / 2) <= 1, "Value was {}".format(x + w / 2)

            size = min(img_w, img_h)

            (xmin, ymin, xmax, ymax) = xywh_to_xyx2y2(x * size, y * size, w * size, h * size)

            new_line = "{} {} {} {} {} {}\n".format(label, prob, xmin, ymin, xmax, ymax)
            content_out.append(new_line)

        # Write updated content to TXT file
        save_name = os.path.splitext(os.path.basename(image))[0] + '.txt'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.writelines(content_out)


def clip_box_to_size(box, size):
    (x, y, w, h) = box # Absolute size
    (im_w, im_h) = size

    # Max length
    l = max(w, h)
    l = min(l, min(im_w, im_h))

    # Make it square, expand a little
    new_x = x
    new_y = y
    new_w = l + 0.075 * min(im_w, im_h)
    new_h = l + 0.075 * min(im_w, im_h)

    # Then clip shape to stay in original image
    xmin, ymin, xmax, ymax = xywh_to_xyx2y2(new_x, new_y, new_w, new_h)
    if xmin < 0:
        new_x = x - xmin
    if xmax >= im_w:
        new_x = x - (xmax - im_w)
    if ymin < 0:
        new_y = y - ymin
    if ymax >= im_h:
        new_y = y - (ymax - im_h)

    return  (new_x, new_y, new_w, new_h)

def read_image_txt_file(file):
    images = []
    with open(file, "r") as f:
        images = [c.strip() for c in f.readlines()]

    return images

def basler3M_calibration_maps(image_size=None):
    """
    Use image_size=None if working with images in original resolution (2048x1536).
    If not, specify the real image size.
    """

    original_img_size = (2048, 1536)

    mtx = np.array([[1846.48412, 0.0,        1044.42589],
                    [0.0,        1848.52060, 702.441180],
                    [0.0,        0.0,        1.0       ]])

    dist = np.array([[-0.19601338, 0.07861078, 0.00182995, -0.00168376, 0.02604818]])

    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(mtx, dist, original_img_size, 0, original_img_size)
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, new_camera_matrix, original_img_size, m1type=cv.CV_32FC1)

    if image_size is not None:
        mapx = cv.resize(mapx, (image_size[0], image_size[1])) * image_size[0] / original_img_size[0]
        mapy = cv.resize(mapy, (image_size[0], image_size[1])) * image_size[1] / original_img_size[1]

    (mapx, mapy) = cv.convertMaps(mapx, mapy, cv.CV_16SC2)

    return (mapx, mapy)


def calibrate(img, mapx, mapy):
    return cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

def RMSE_opt_flow(txt_file):
    images = []
    with open(txt_file, "r") as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    first_image = cv.imread(images[0])
    (img_h, img_w) = first_image.shape[:2]

    xmin = int(img_w * 0.2)
    xmax = int(img_w * 0.9)
    ymin = int(img_h * 0.1)
    ymax = int(img_h * 0.9)

    base_mask = np.full(first_image.shape[:2], False)
    base_mask[ymin:ymax:, xmin:xmax] = True

    opflow = None

    X, Y = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))

    RMSEs_X = [0]
    RMSEs_Y = [0]

    for image in images[1:]:
        second_image = cv.imread(image)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        egi = np.uint8(egi_mask(first_image) * 255)
        egi = cv.dilate(egi, kernel, iterations=5)
        egi = egi.astype(np.bool)

        mask = ~egi & base_mask

        opflow, tmp = optical_flow(first_image, second_image,
            prev_opt_flow=opflow)

        dX, dY = opflow[..., 0], opflow[..., 1]

        coeffs_X = fit_plane(X, Y, dX)
        coeffs_Y = fit_plane(X, Y, dY)

        coeffs_X_2 = fit_plane(X, Y, dX, mask)
        coeffs_Y_2 = fit_plane(X, Y, dY, mask)

        H_x = coeffs_X[0] * X + coeffs_X[1] * Y + coeffs_X[2]
        H_y = coeffs_Y[0] * X + coeffs_Y[1] * Y + coeffs_Y[2]

        H_x_2 = coeffs_X_2[0] * X + coeffs_X_2[1] * Y + coeffs_X_2[2]
        H_y_2 = coeffs_Y_2[0] * X + coeffs_Y_2[1] * Y + coeffs_Y_2[2]

        RMSE_x = np.sqrt(np.mean(np.square(H_x - H_x_2)))
        RMSE_y = np.sqrt(np.mean(np.square(H_y - H_y_2)))

        RMSEs_X.append(RMSE_x)
        RMSEs_Y.append(RMSE_y)

        print(RMSE_x, RMSE_y)

        first_image = tmp

    plt.plot(RMSEs_X, "b", label="X")
    plt.plot(RMSEs_Y, "r", label="Y")
    plt.legend()
    plt.xlabel("Frame number")
    # plt.ylabel("RMSE")
    plt.title("Optical Flow Mask Comparison - Planar Opt Flow")
    plt.show()

def another_optical_flow_test(txt_file,
    name="opt_flow.txt", masking_border=False, mask_egi=False
):
    images = []
    with open(txt_file, "r") as f:
        images = f.readlines()
    images = [image.strip() for image in images]

    past_image = cv.imread(images[0])

    optical_flow_fn = OpticalFlow(past_image.shape[:2],
        mask_border=masking_border, mask_egi=mask_egi)

    flows = []

    for i, image in enumerate(images[1:]):
        current_image = cv.imread(image)

        # OF from current image to previous image
        eq_x, eq_y = optical_flow_fn.displacement_plane_equation(current_image, past_image)
        flows.append(BivariateFunction(eq_x, eq_y))

        past_image = current_image

def draw_tracked_confidence_ellipse(tracks_dict, save_dir=None):
    save_dir = "save/aggr_tracking_ellispe/" if not save_dir else save_dir
    create_dir(save_dir)
    
    for (image, tracks) in tracks_dict.items():
        img = cv.imread(image)

        cmap = cm.get_cmap(cm.rainbow)(np.linspace(0.0, 1.0, len(tracks)))[:, :3]
        for i, track in enumerate(tracks):
            color = cmap[i] * 255
            for box in track:
                (x, y, _, _) = box.getAbsoluteBoundingBox(BBFormat.XYC)

                cv.circle(img,
                    center=(int(x), int(y)), radius=2,
                    color=color, thickness=cv.FILLED)

            (ex, ey, w, h, a) = track.confidence_ellipse(n_std=1)

            cv.ellipse(img,
                center=(int(ex), int(ey)), axes=(int(w), int(h)), angle=a,
                startAngle=0, endAngle=360, color=color, thickness=2)

        save_name = os.path.join(save_dir, os.path.basename(image))
        cv.imwrite(save_name, img)

def get_perspective_dataset(path, save_dir=None):
    images = files_with_extension(path, ".jpg")

    if save_dir is None:
        save_dir = "perspective_dataset/"
    create_dir(save_dir)

    transform = imtf.get_transformation((632, 632),
        rx=imtf.radians(5))  # Y axis = rx
        # ry=imtf.radians(0), dx=0, dz=0)

    for image in images:
        img = cv.imread(image)
        warped = imtf.warp_perspective(img, transform)
        save_name = os.path.join(save_dir, os.path.basename(image))
        cv.imwrite(save_name, warped)

def normalized_stem_boxes(boxes,
	ratio=7.5/100, labels=["haricot_tige", "mais_tige", "poireau_tige"]
):
	normalized_boxes = BoundingBoxes()

	for box in boxes:
		if box.getClassId() in labels:
			normalized_boxes.append(box.normalized(ratio))
		else:
			normalized_boxes.append(box)

	return normalized_boxes

def create_image_list_file(folder, save_dir=None):
    save_dir = save_dir or folder
    create_dir(save_dir)
    images = sorted(files_with_extension(folder, ".jpg"))
    with open(os.path.join(save_dir, "image_list.txt"), "w") as f:
        for image in images:
            f.write(image + "\n")


if __name__ == "__main__":
    # folder = "/home/deepwater/Downloads/operose_test_maize"
    # folder = "/home/deepwater/Downloads/operose_test"
    # folder="/media/deepwater/DATA/Shared/Louis/datasets/mais_debug_montoldre_2"
    # folder="/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2"

    folder = "/media/deepwater/DATA/Shared/Louis/datasets/tache_detection/maize (copy)"
    img_list = os.path.join(folder, "image_list.txt")
    create_image_list_file(folder)
    optflow_file = os.path.join(folder, "optical_flow.txt")
    OpticalFlow.generate(img_list, "optical_flow.txt", save_dir=folder, masking_border=True, mask_egi=True)
    boxes = Parser.parse_json_folder(folder, classes={"maize", "maize_stem"})
    print(boxes)

    # image_list_file = os.path.join(folder, "image_list.txt")
    # images = sorted(read_image_txt_file(image_list_file))

    # flow = OpticalFlowLK()
    # prev = cv.imread(images[0])
    # prev = cv.resize(prev, (512, 512))

    # for image in images[1:]:
    #     next = cv.imread(image)
    #     next = cv.resize(next, (512, 512))
    #     (dx, dy) = flow(next, prev)
    #     print(os.path.basename(image), dx, dy)

    #     prev = next

    # get_perspective_dataset("/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2")
    #
    # images = read_image_txt_file("data/haricot_sequential.txt")
    # create_dir("tmp/")
    # for image in images:
    #     img = cv.imread(image)
    #     img = egi_mask(img).astype(np.uint8) * 255
    #     cv.imwrite(f"tmp/{os.path.basename(image)}", img)

    # another_optical_flow_test("data/haricot_debug_long_2.txt",
    #     name="data/test_opt_flow.txt", masking_border=True, mask_egi=True)


    # img1 = cv.imread("/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2/im_03331.jpg")
    # img2 = cv.imread("/media/deepwater/DATA/Shared/Louis/datasets/haricot_debug_montoldre_2/im_03332.jpg")
    #
    # of, _ = optical_flow(img1, img2)
    # dx = of[..., 0]
    # print(dx.shape)
    # plt.imshow(dx)
    # plt.show()

    # colors = [(1, 0, 0), (0.5, 0, 0), (0.25, 0, 0)]
    #
    # mu = [10, 100]
    # scales = [3, 5]
    # cov_mat = [
    #     [0.9, -0.4],
    #     [0.1, -0.6]
    # ]
    #
    # (x, y) = get_correlated_dataset(100, cov_mat, mu, scales)
    #
    # figure = plt.figure()
    # axis = plt.subplot(111)
    # axis.axis("equal")
    # for n in range(1, 4):
    #     (ex, ey, w, h, angle) = confidence_ellipse(x, y, n_std=n)
    #     print("x: {:.4}, y: {:.4}, w: {:.4}, h: {:.4}, angle: {:.4}".format(ex, ey, w, h, angle))
    #     ellipse = Ellipse(xy=(ex, ey), width=w, height=h, angle=angle, color=colors[n-1])
    #     ellipse.set_facecolor("none")
    #     axis.add_artist(ellipse)
    # plt.scatter(x, y)
    # plt.show()

    # txt_file = "data/haricot_debug_long_2.txt"

    # opt_flow_plane(txt_file)
    # generate_opt_flow(txt_file, "opt_flow_mean_none.txt", masking_border=False, mask_egi=False)
    # generate_opt_flow(txt_file, "opt_flow_mean_all.txt", masking_border=True, mask_egi=True)
    # generate_opt_flow(txt_file, "opt_flow_mean_egi.txt", masking_border=False, mask_egi=True)
    # generate_opt_flow(txt_file, "opt_flow_mean_border.txt", masking_border=True, mask_egi=False)
    #
    # # RMSE_opt_flow(txt_file)
    #
    # with open("opt_flow_mean_none.txt", "r") as f:
    #     content = f.readlines()
    #     content = np.array([c.strip().split() for c in content], dtype=np.float)
    #     dx1, dy1 = content[:, 0], content[:, 1]
    #
    # with open("opt_flow_mean_all.txt", "r") as f:
    #     content = f.readlines()
    #     content = np.array([c.strip().split() for c in content], dtype=np.float)
    #     dx2, dy2 = content[:, 0], content[:, 1]
    #
    # with open("opt_flow_mean_egi.txt", "r") as f:
    #     content = f.readlines()
    #     content = np.array([c.strip().split() for c in content], dtype=np.float)
    #     dx3, dy3 = content[:, 0], content[:, 1]
    #
    # with open("opt_flow_mean_border.txt", "r") as f:
    #     content = f.readlines()
    #     content = np.array([c.strip().split() for c in content], dtype=np.float)
    #     dx4, dy4 = content[:, 0], content[:, 1]
    #
    # plt.plot(dx1, label="None")
    # plt.plot(dx2, label="All")
    # plt.plot(dx3, label="EGI")
    # plt.plot(dx4, label="Border")
    # plt.legend()
    # plt.title("Mean Opt flow in X axis w.r.t masking")
    # plt.show()

    # err_dx = np.sqrt(np.mean(np.square(dx1 - dx2)))
    # print("RMSE X: {}".format(err_dx))
    #
    # err_dy = np.sqrt(np.mean(np.square(dy1 - dy2)))
    # print("RMSE Y: {}".format(err_dy))
    #
    # plt.plot(dx1, "b", label="Mean Opt Flow")
    # plt.plot(dx2, "r", label="Planar Center Opt Flow")
    # plt.legend()
    # plt.title("Optical Flow")
    # plt.show()
