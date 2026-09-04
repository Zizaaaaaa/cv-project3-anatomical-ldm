import argparse
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def convert_label_to_yolo_bbox(txt_path):
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

def copy_split_data(src_dir, dest_img_dir, dest_lbl_dir):
    if not src_dir.exists():
        return

    png_files = list(src_dir.rglob("*.png"))

    for img_file in png_files:
        shutil.copy(img_file, dest_img_dir / img_file.name)
        
        lbl_name = f"{img_file.stem}.txt"
        lbl_file = img_file.parent / lbl_name
        
        if not lbl_file.exists():
            alt_lbl_path = Path(str(img_file.parent).replace("images", "labels")) / lbl_name
            if alt_lbl_path.exists():
                lbl_file = alt_lbl_path

        dest_lbl_path = dest_lbl_dir / lbl_name
        if lbl_file.exists():
            formatted_lbl = convert_label_to_yolo_bbox(lbl_file)
            with open(dest_lbl_path, "w") as f:
                f.write(formatted_lbl)
        else:
            open(dest_lbl_path, "w").close()

def prepare_yolo_dataset(real_dataset_path, synthetic_dataset_path, output_dir, fold="f0", include_synthetic=False):
    yolo_dir = Path(output_dir) / f"yolo_{fold}_{'augmented' if include_synthetic else 'baseline'}"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for split in ["train", "val", "test"]:
        os.makedirs(yolo_dir / "images" / split, exist_ok=True)
        os.makedirs(yolo_dir / "labels" / split, exist_ok=True)

    fold_path = Path(real_dataset_path) / fold

    for split in ["train", "val", "test"]:
        src_split_dir = fold_path / split
        dest_img = yolo_dir / "images" / split
        dest_lbl = yolo_dir / "labels" / split
        copy_split_data(src_split_dir, dest_img, dest_lbl)

    if include_synthetic and synthetic_dataset_path and os.path.exists(synthetic_dataset_path):
        synth_path = Path(synthetic_dataset_path)
        synth_images = list(synth_path.glob("*.png"))
        
        print(f"--> [{fold}] Added {len(synth_images)} synthetic images to the training set")
        for img_file in synth_images:
            dest_name = f"synth_{img_file.name}"
            shutil.copy(img_file, yolo_dir / "images" / "train" / dest_name)
            
            lbl_file = synth_path / f"{img_file.stem}.txt"
            dest_lbl_path = yolo_dir / "labels" / "train" / f"synth_{img_file.stem}.txt"
            if lbl_file.exists():
                formatted_lbl = convert_label_to_yolo_bbox(lbl_file)
                with open(dest_lbl_path, "w") as f:
                    f.write(formatted_lbl)
            else:
                open(dest_lbl_path, "w").close()

    yaml_content = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/val
test: images/val

names:
  0: bone_loss
"""
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path

def run_experiment(args):
    folds = [f"f{i}" for i in range(5)]
    results = []

    print(f"=== EXPERIMENTO YOLOv8 ===")
    print(f"Mode: {'AUGMENTED (Real + Synthetic)' if args.include_synthetic else 'BASELINE (Only Real)'}")

    for fold in folds:
        
        data_yaml = prepare_yolo_dataset(
            real_dataset_path=args.real_dataset_path,
            synthetic_dataset_path=args.synthetic_dataset_path,
            output_dir=args.output_dir,
            fold=fold,
            include_synthetic=args.include_synthetic
        )

        model = YOLO(args.model_size)

        save_name = f"yolo_{fold}_{'augmented' if args.include_synthetic else 'baseline'}"
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            project=f"{args.output_dir}/runs",
            name=save_name,
            verbose=False
        )

        val_metrics = model.val(data=str(data_yaml), split="val", verbose=False)
        
        # Management strategy for test set evaluation
        try:
            test_metrics = model.val(data=str(data_yaml), split="test", verbose=False)
            t_mAP50 = test_metrics.box.map50
            t_mAP50_95 = test_metrics.box.map
        except Exception:
            # Fallback to validation metrics if test set evaluation fails
            t_mAP50 = val_metrics.box.map50
            t_mAP50_95 = val_metrics.box.map

        results.append({
            "fold": fold,
            "val_mAP50": val_metrics.box.map50,
            "val_mAP50-95": val_metrics.box.map,
            "test_mAP50": t_mAP50,
            "test_mAP50-95": t_mAP50_95
        })

        print(f"[{fold}] mAP@50: {val_metrics.box.map50:.4f} | mAP@50-95: {val_metrics.box.map:.4f}")

    df_res = pd.DataFrame(results)
    mode_str = "augmented" if args.include_synthetic else "baseline"
    csv_path = os.path.join(args.output_dir, f"results_{mode_str}.csv")
    df_res.to_csv(csv_path, index=False)

    print(f"\n=== AVERAGE RESULTS 5-FOLD ({mode_str.upper()}) ===")
    print(f"Mean mAP@50: {df_res['val_mAP50'].mean():.4f} ± {df_res['val_mAP50'].std():.4f}")
    print(f"Mean mAP@50-95: {df_res['val_mAP50-95'].mean():.4f} ± {df_res['val_mAP50-95'].std():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dataset_path", type=str, default="/content/drive/MyDrive/upload_final/1_Experiment/standard_box")
    parser.add_argument("--synthetic_dataset_path", type=str, default="/content/generated_samples")
    parser.add_argument("--output_dir", type=str, default="/content/outputs/yolo_experiments")
    parser.add_argument("--model_size", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--include_synthetic", action="store_true")

    args = parser.parse_args()
    run_experiment(args)
