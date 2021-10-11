from pathlib import Path
import numpy as np
import subprocess


DARKNET_PATH = Path("/home/deepwater/Documents/darknet/")
RESULTS_PATH = DARKNET_PATH / "results/"
YOLO_PATH =  RESULTS_PATH / "yolov4-tiny_res_544/"


def main():
    confidences = np.linspace(0.0, 1.0, 21)
    preds_dir = YOLO_PATH / "predictions/"
    gts_dir = DARKNET_PATH / f"data/database_13.0_norm/val/"

    for confidence in confidences:
        subprocess.run(["python", "evaluate_dist_f1.py",
            str(gts_dir), "--dets_dir", str(preds_dir),
            "--conf_threshold", f"{confidence:.2}", 
            "--save_csv", str(YOLO_PATH / f"f1_conf_{confidence:.2}.csv")])


if __name__ == "__main__":
    main()