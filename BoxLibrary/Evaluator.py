import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from .BoundingBox import BoundingBox
from .BoundingBoxes import BoundingBoxes
from .utils import *

import rich
from rich import print as rprint
from rich.table import Table, Column

from collections import defaultdict


class Evaluator:
    cocoThresholds = [thresh / 100 for thresh in range(50, 100, 5)]

    def GetPascalVOCMetrics(self, 
        boxes, 
        thresh=0.5, 
        method=EvaluationMethod.IoU, 
        labels=None
    ):
        ret = {}
        boxesByLabels = dictGrouping(boxes, key=lambda box: box.getClassId())
        labels = sorted(labels or boxesByLabels.keys())

        for label in labels:
            boxesByDetectionMode = dictGrouping(
                boxesByLabels[label],
                key=lambda box: box.getBBType())

            groundTruths = dictGrouping(
                boxesByDetectionMode[BBType.GroundTruth],
                key=lambda box: box.getImageName())

            detections = sorted(
                boxesByDetectionMode[BBType.Detected],
                key=lambda box: box.getConfidence(),
                reverse=True)

            TP = np.repeat(False, len(detections))
            npos = len(boxesByDetectionMode[BBType.GroundTruth])
            accuracies = []
            visited = {k: np.repeat(False, len(v)) for k, v in groundTruths.items()}

            for (i, detection) in enumerate(detections):
                imageName = detection.getImageName()
                associatedGts = groundTruths[imageName]

                if method == EvaluationMethod.IoU:
                    maxIoU = 0

                    for j, gt in enumerate(associatedGts):
                        iou = detection.iou(gt)

                        if iou > maxIoU:
                            maxIoU = iou
                            jmax = j

                    if maxIoU > thresh and not visited[imageName][jmax]:
                        visited[imageName][jmax] = True
                        TP[i] = True
                        accuracies.append(maxIoU)

                if method == EvaluationMethod.Distance:
                    minDist = sys.float_info.max
                    minImgSize = min(detection.getImageSize())
                    normThresh = thresh * minImgSize

                    for (j, gt) in enumerate(associatedGts):
                        dist = detection.distance(gt)

                        if dist < minDist:
                            minDist = dist
                            jmin = j

                    if minDist < normThresh and not visited[imageName][jmin]:
                        visited[imageName][jmin] = True
                        TP[i] = True
                        accuracies.append(minDist / minImgSize)

            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum([not tp for tp in TP])
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            total_tp = sum(TP)
            total_fp = len(TP) - total_tp

            ap, *_ = Evaluator.CalculateAveragePrecision(rec, prec)

            ret[label] = {
                "precision": prec,
                "recall": rec,
                "AP": ap,
                "threshold": thresh,
                "total positives": npos,
                "total detections": len(TP),
                "total TP": total_tp,
                "total FP": total_fp,
                "accuracies": np.array(accuracies),
                "evaluation method": method
            }

        return ret

    def getCocoMetrics(self, boundingBoxes):
        return [self.GetPascalVOCMetrics(boundingBoxes, thresh)
            for thresh in self.cocoThresholds]

    def getAP(self, boundingBoxes, thresh=0.5, method=EvaluationMethod.IoU):
        AP = [res["AP"]
            for res in self.GetPascalVOCMetrics(boundingBoxes, thresh, method).values()]
        return sum(AP) / len(AP) if AP else 0.0

    def getCocoAP(self, boundingBoxes):
        AP = [self.getAP(boundingBoxes, thresh)
            for thresh in self.cocoThresholds]
        return sum(AP) / len(AP) if AP else 0.0

    def printAPs(self, boxes):
        APs = [self.getAP(boxes, thresh)
            for thresh in self.cocoThresholds]
        cocoAP = sum(APs) / len(APs) if APs else 0.0

        print("mAP@.50: {:.2%}".format(APs[0]))
        print("mAP@.75: {:.2%}".format(APs[5]))
        print("coco AP: {:.2%}".format(cocoAP))

    def printCOCOAPByClass(self, boxes, labels=None):
        results = [self.GetPascalVOCMetrics(boxes, t, labels=labels)
            for t in self.cocoThresholds]

        aps = defaultdict(list)
        for res in results:
            for label, res_label in res.items():
                aps[label].append(res_label["AP"])
        
        coco_aps = {l: sum(r) / len(r) if r else 0.0  for l, r in aps.items()}
        aps_50 = {l: a[0] for l, a in aps.items()}
        aps_75 = {l: a[5] for l, a in aps.items()}
        
        coco_ap = sum(coco_aps.values()) / len(coco_aps)
        ap_50 = sum(aps_50.values()) / len(aps_50)
        ap_75 = sum(aps_75.values()) / len(aps_75)

        table = Table(show_footer=True)
        table.add_column("Label", "Total")
        table.add_column("COCO AP", f"{coco_ap:.2%}", justify="right")
        table.add_column("AP 50", f"{ap_50:.2%}", justify="right")
        table.add_column("AP 75", f"{ap_75:.2%}", justify="right")

        for (label, ap), ap50, ap75 in zip(
            coco_aps.items(), aps_50.values(), aps_75.values()
        ):
            table.add_row(label, f"{ap:.2%}", f"{ap50:.2%}", f"{ap75:.2%}")

        rprint(table)

    def printAPsByClass(self, boxes, thresh=0.5, method=EvaluationMethod.IoU):
        metrics = self.GetPascalVOCMetrics(boxes, thresh, method)
        tot_tp, tot_fp, tot_npos, accuracy = 0, 0, 0, 0
        accuracies = []
        print("AP@{} by class:".format(thresh))

        for label, metric in metrics.items():
            AP = metric["AP"]
            totalPositive = metric["total positives"]
            totalDetections = metric["total detections"]
            TP = metric["total TP"]
            FP = metric["total FP"]
            accuracy += sum(metric["accuracies"])
            accuracies.extend(metric["accuracies"])
            tot_tp += TP
            tot_fp += FP
            tot_npos += totalPositive

            print("  {:<10} - AP: {:.2%}  npos: {}  nDet: {}  TP: {}  FP: {}".format(label, AP, totalPositive, totalDetections, TP, FP))

        recall = tot_tp / tot_npos
        precision = tot_tp / (tot_tp + tot_fp)
        f1 = 2 * recall * precision / (recall + precision)
        accuracy /= tot_tp

        std = np.std(accuracies)
        err = std / np.sqrt(len(accuracies))

        print("Global stats: ")
        print("  recall: {:.2%}, precision: {:.2%}, f1: {:.2%}, acc: {:.2%}, err_acc: {:.2%}".format(recall, precision, f1, accuracy, err))

        return (recall, precision, f1)

    def printF1ByClass(self, boxes, threshold=25/100, method=EvaluationMethod.Distance):
        metrics = self.GetPascalVOCMetrics(boxes, threshold, method)

        description = "Label     |npos|ndet|rec    |prec   |f1     |acc    |acc_err\n"
        description += "------------------------------------------------------------\n"
        tot_tp, tot_fp, tot_npos = 0, 0, 0
        tot_accuracies = []

        for label, metric in metrics.items():
            tp, fp = metric["total TP"], metric["total FP"]
            recall, precision = metric["recall"], metric["precision"]
            npos, ndet = metric["total positives"], metric["total detections"]
            accuracies = metric["accuracies"]
            precision = tp / ndet if ndet != 0 else 1 if npos == 0 else 0
            recall = tp / npos if npos != 0 else 1 if ndet == 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
            accuracy = accuracies.mean()
            std_err = np.std(accuracies) / np.sqrt(len(accuracies))
            description += f"{label:10}|{npos:4}|{ndet:4}|{recall:7.2%}|{precision:7.2%}|{f1:7.2%}|{accuracy:7.2%}|{std_err:7.2%}\n"

            tot_tp += tp
            tot_fp += fp
            tot_npos += npos
            tot_accuracies.extend(accuracies)

        tot_ndet = tot_tp + tot_fp
        tot_precision = tot_tp / tot_npos
        tot_recall = tot_tp / tot_ndet
        tot_f1 = 2 * tot_precision * tot_recall / (tot_precision + tot_recall)
        tot_accuracies = np.array(tot_accuracies)
        tot_accuracy = tot_accuracies.mean()
        tot_std_err = np.std(tot_accuracies) / np.sqrt(len(tot_accuracies))
        description += "------------------------------------------------------------\n"
        description += f"{'total':10}|{tot_npos:4}|{tot_ndet:4}|{tot_recall:7.2%}|{tot_precision:7.2%}|{tot_f1:7.2%}|{tot_accuracy:7.2%}|{tot_std_err:7.2%}"

        print(description)

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = [0]
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = [0]
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
            
        ii = [i + 1 for i in range(len(mrec) - 1) if mrec[1:][i] != mrec[0:-1][i]]
        ap = sum(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) for i in ii)

        return [ap, mpre[0:-1], mrec[0:len(mpre) - 1], ii]
