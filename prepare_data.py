import os
import math
import random
from PIL import Image

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def get_kp(parts, index, img_w, img_h):
    base_idx = 5 + index * 3
    if base_idx + 2 >= len(parts):
        return None
    x = float(parts[base_idx])
    y = float(parts[base_idx + 1])
    v = float(parts[base_idx + 2])
    if v == 0.0:
        return None
    return x * img_w, y * img_h

def prepare_data():
    base_dir = "data/UPLOAD_FINAL/0_Baseline"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    output_dir = "processed_patches_stage4"
    
    os.makedirs(output_dir, exist_ok=True)
    
    VALID_CLASSES = [0, 1, 2] #single, double, triple root
    
    # soglia abbassata a da 0.5 a 0.33 per aumentare il numero di casi severi
    BONE_LOSS_THRESHOLD = 0.33 
    
    # numero di patch da creare per ogni caso severo (per aumentare la variabilità dei dati)
    AUGMENTATIONS_PER_PATCH = 10
    
    patch_count = 0
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
            
        base_name = os.path.splitext(label_file)[0]
        image_path = os.path.join(images_dir, f"{base_name}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found for label {label_file}.")
            continue
            
        label_path = os.path.join(labels_dir, label_file)        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            continue
            
        img = None
        img_w, img_h = 0, 0
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])       
            if class_id not in VALID_CLASSES:
                continue
                
            if len(parts) < (5 + 7 * 3):
                continue
            

            if img is None:
                img = Image.open(image_path).convert("RGB")
                img_w, img_h = img.size

            l_cej = get_kp(parts, 0, img_w, img_h)
            l_bl = get_kp(parts, 1, img_w, img_h)
            l_apex = get_kp(parts, 2, img_w, img_h)
            r_cej = get_kp(parts, 3, img_w, img_h)
            r_bl = get_kp(parts, 4, img_w, img_h)
            r_apex = get_kp(parts, 5, img_w, img_h)
            c_apex = get_kp(parts, 6, img_w, img_h)

            actual_l_apex = l_apex if l_apex is not None else c_apex
            actual_r_apex = r_apex if r_apex is not None else c_apex

            max_bone_loss_pct = 0.0

            if l_cej and l_bl and actual_l_apex:
                root_length_l = calculate_distance(l_cej, actual_l_apex)
                bone_loss_l = calculate_distance(l_cej, l_bl)
                if root_length_l > 0:
                    pct_l = bone_loss_l / root_length_l
                    max_bone_loss_pct = max(max_bone_loss_pct, pct_l)

            if r_cej and r_bl and actual_r_apex:
                root_length_r = calculate_distance(r_cej, actual_r_apex)
                bone_loss_r = calculate_distance(r_cej, r_bl)
                if root_length_r > 0:
                    pct_r = bone_loss_r / root_length_r
                    max_bone_loss_pct = max(max_bone_loss_pct, pct_r)

            is_severe = max_bone_loss_pct > BONE_LOSS_THRESHOLD

            if is_severe:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                cx = x_center * img_w
                cy = y_center * img_h
                w = width * img_w
                h = height * img_h
                
                loss_str = int(max_bone_loss_pct * 100)
                
                # Generate multiple augmented patches per severe case
                for aug_idx in range(AUGMENTATIONS_PER_PATCH):
                    #randomly scale
                    scale_factor = random.uniform(0.9, 1.3)
                    aug_w = w * scale_factor
                    aug_h = h * scale_factor
                    
                    #randomly shift a little the center
                    shift_x = random.uniform(-0.1, 0.1) * aug_w
                    shift_y = random.uniform(-0.1, 0.1) * aug_h
                    
                    aug_cx = cx + shift_x
                    aug_cy = cy + shift_y
                    
                    x_min = int(aug_cx - aug_w / 2)
                    y_min = int(aug_cy - aug_h / 2)
                    x_max = int(aug_cx + aug_w / 2)
                    y_max = int(aug_cy + aug_h / 2)
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_w, x_max)
                    y_max = min(img_h, y_max)
                    
                    # Skip if crop area is too small
                    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                        continue
                        
                    patch = img.crop((x_min, y_min, x_max, y_max))
                    patch_resized = patch.resize((256, 256), Image.Resampling.LANCZOS)
                    
                    output_filename = f"{base_name}_patch_{i}_loss_{loss_str}_aug_{aug_idx}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    patch_resized.save(output_path)
                    
                    patch_count += 1

                
    print(f"Extraction complete! Saved {patch_count} augmented severe patches to {output_dir}/")

if __name__ == "__main__":
    prepare_data()
