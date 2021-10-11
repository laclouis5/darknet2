from pathlib import Path
import numpy as np


def main():
    save_dir = Path("all_f1_conf.csv")
    confidences = np.linspace(0.0, 1.0, 21)
    concat = []

    for confidence in confidences:
        path = Path(f"results/yolov4-tiny_res_544/f1_conf_{confidence:.2}.csv")

        content = path.read_text().splitlines()[1:]
        concat += content

    save_dir.write_text("\n".join(concat))


if __name__ == "__main__":
    main()