import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm
import shutil


DARKNET_DIR = Path("/home/deepwater/Documents/darknet/")
CFG_PATH = DARKNET_DIR / "cfg/yolov4-tiny.cfg"
MODEL_PATH = DARKNET_DIR / "models/yolov4-tiny.conv.29"


def train(db_path: Path, scale: float, save_dir: Path):
    save_dir = DARKNET_DIR / f"results/yolov4-tiny_norm_{scale:.2}/"
    db_path = DARKNET_DIR / f"data/database_13.0_norm{scale:.2}/"
    meta_path = db_path / "obj.data"
    names_path = db_path / "obj.names"
    best_model = DARKNET_DIR / "backup/yolov4-tiny_best.weights"

    subprocess.run(["./darknet", "detector", "train",
        f"{meta_path}", f"{CFG_PATH}", f"{MODEL_PATH}",
        "-map"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    save_dir.mkdir(exist_ok=True)

    shutil.copy(f"{DARKNET_DIR / 'chart.png'}", f"{save_dir / 'chart.png'}")
    shutil.copy(f"{best_model}", f"{save_dir / best_model.name}")
    shutil.copy(f"{CFG_PATH}", f"{save_dir / CFG_PATH.name}")
    shutil.copy(f"{meta_path}", f"{save_dir / meta_path.name}")
    shutil.copy(f"{names_path}", f"{save_dir / names_path.name}")


def main():
    scales = np.exp(np.linspace(np.log(0.01), np.log(0.5), 20))
    all = []

    for scale in tqdm(scales):
        save_dir = DARKNET_DIR / f"results/yolov4-tiny_norm_{scale:.2}/"
        db_path = DARKNET_DIR / f"data/database_13.0_norm{scale:.2}/"
        
        # train(db_path, scale, save_dir)

        # subprocess.run(["python", "detect_and_save.py", 
        #     str(db_path / "val/"), str(save_dir), str(save_dir / "predictions")], 
        #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        subprocess.run(["python", "evaluate_dist_f1.py",
            str(db_path / "val/"), "--dets_dir", str(save_dir / "predictions/"),
            "--conf_threshold", "0.25",
            "--save_csv", f"{save_dir / f'evaluation_f1_norm_{scale:.2}'}"])

        path = save_dir / f'evaluation_f1_norm_{scale:.2}'
        content = path.read_text().splitlines()[1:]
        all.append("\n".join(content))

    (DARKNET_DIR / "all_f1_eval_norm.csv").write_text("\n".join(all))


if __name__ == "__main__":
    main()