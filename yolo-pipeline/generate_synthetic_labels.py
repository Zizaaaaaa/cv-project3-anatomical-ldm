import os
import argparse
from pathlib import Path
import numpy as np

def convert_label_to_yolo_bbox(txt_path: Path) -> str:
    if not txt_path.exists():
        return ""

    lines = []
    with open(txt_path, "r") as f:
        content = f.readlines()

    for line in content:
        line_str = line.strip()
        if not line_str:
            continue
        
        parts = line_str.split()
        if len(parts) == 5:
            lines.append(line_str)
            continue
            
        try:
            vals = [float(p) for p in parts]
            if len(vals) % 2 == 1:
                cls_id = int(vals[0])
                coords = np.array(vals[1:]).reshape(-1, 2)
            else:
                cls_id = 0
                coords = np.array(vals).reshape(-1, 2)

            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

            x_min, x_max = max(0.0, x_min), min(1.0, x_max)
            y_min, y_max = max(0.0, y_min), min(1.0, y_max)

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            w = x_max - x_min
            h = y_max - y_min

            if w > 0 and h > 0:
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        except Exception:
            pass

    return "\n".join(lines)

def process_synthetic_folder(synth_dir: str):
    synth_path = Path(synth_dir)
    txt_files = list(synth_path.glob("*.txt"))
    
    print(f"Processing {len(txt_files)} label files in {synth_dir}...")
    for txt_file in txt_files:
        formatted = convert_label_to_yolo_bbox(txt_file)
        with open(txt_file, "w") as f:
            f.write(formatted)
    print("Synthetic label conversion completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, default="/content/generated_samples")
    args = parser.parse_args()
    process_synthetic_folder(args.synthetic_dir)
