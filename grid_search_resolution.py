import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import os


darknet_dir = Path("/home/deepwater/Documents/darknet/")
data_dir = darknet_dir / "data"
results_dir = darknet_dir / "results"
cfg_dir = darknet_dir / "cfg"
backup_dir = darknet_dir / "backup"
val_dir = data_dir / "database_13.0_norm/val"

obj_path = data_dir / "obj.data"
names_path = data_dir / "obj.names"

resolutions = [i*32 for i in range(10, 31)][::-1]


class CFGParser:
    """Hacky, may not be reliable, double check that it worked properly."""
    def __init__(self, cfg_file_path: Path) -> None:
        assert cfg_file_path.is_file()

        data = []   
        content = cfg_file_path.read_text()
        sections = content.split("[")

        for section in sections:
            if section == "": continue
            name, pairs = section.split("]")
            key_value_pairs = {}
            for pair in pairs.split("\n"):
                if "=" in pair:
                    k, v = pair.split("=")
                    key_value_pairs[k.strip()] = v.strip()
            data.append((name, key_value_pairs))

        self.data = data

    def set_net_size(self, size: "tuple[int, int]"):
        if isinstance(size, tuple):
            width, height = size
        else:
            width = size
            height = size

        assert width > 0
        assert height > 0
 
        self.data[0][1]["width"] = width
        self.data[0][1]["height"] = height

    def set_bs_subdiv(self, batch_size: int, subdivisions: int):
        assert batch_size > 0
        assert subdivisions > 0

        self.data[0][1]["batch"] = batch_size
        self.data[0][1]["subidivisions"] = subdivisions

    def string(self) -> str:
        return "\n\n".join(
            "\n".join((f"[{name}]", *(f"{k}={v}" for k, v in pairs.items())))
            for name, pairs in self.data)

    def write(self, path: Path):
        path.write_text(self.string())


def save(resolution: int, save_dir: Path):
    save_dir.mkdir()

    chart_path = darknet_dir / f"chart_yolov4-tiny_res_{resolution}.png"
    best_path = backup_dir / f"yolov4-tiny_res_{resolution}_best.weights"
    cfg_path = cfg_dir / f"yolov4-tiny_res_{resolution}.cfg"

    shutil.copy(str(chart_path), save_dir / str(chart_path.name))
    shutil.copy(str(best_path), save_dir / str(best_path.name))
    shutil.copy(str(obj_path), save_dir / str(obj_path.name))
    shutil.copy(str(names_path), save_dir / str(names_path.name))
    shutil.copy(str(cfg_path), save_dir / str(cfg_path.name))


def train_resolution(resolution: int):
    assert resolution > 0
    save_dir = results_dir / f"yolov4-tiny_res_{resolution}"
    cfg = cfg_dir / f"yolov4-tiny_res_{resolution}.cfg"

    subprocess.run(["./darknet", "detector", "train", 
        "data/obj.data", cfg ,"models/yolov4-tiny.conv.29", "-map"])

    save(resolution, save_dir)


def main():
    os.chdir(darknet_dir)
    # parser = CFGParser(cfg_path)

    for resolution in tqdm(resolutions):
        # parser.set_net_size(resolution)
        # parser.set_bs_subdiv(64, 8)
        # parser.write(cfg_dir / f"yolov4-tiny_res_{resolution}.cfg")

        # train_resolution(resolution)

        # subprocess.run(["python", "detect_and_save.py", "data/val/", f"results/yolov4-tiny_res_{resolution}", f"results/yolov4-tiny_res_{resolution}/predictions/"])

        subprocess.run(["python", "evaluate_dist_f1.py",
            str(val_dir), "--dets_dir", str(f"results/yolov4-tiny_res_{resolution}/predictions/"),
            "--conf_threshold", "0.5", "--save_csv", str(f"results/yolov4-tiny_res_{resolution}/evaluation_f1_res_{resolution}_50.csv")])

if __name__ == "__main__":
    main()