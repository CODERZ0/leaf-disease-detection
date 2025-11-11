import os, shutil, random
from tqdm import tqdm

def split_folder(src_dir, dest_dir, val_frac=0.2, test_frac=0.1, seed=42):
    random.seed(seed)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    os.makedirs(dest_dir, exist_ok=True)
    for split in ['train','val','test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    for cls in tqdm(classes, desc="Classes"):
        cls_src = os.path.join(src_dir, cls)
        images = [f for f in os.listdir(cls_src) if os.path.isfile(os.path.join(cls_src, f))]
        random.shuffle(images)
        n = len(images)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_imgs = images[:n_test]
        val_imgs = images[n_test:n_test + n_val]
        train_imgs = images[n_test + n_val:]

        for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            cls_dest = os.path.join(dest_dir, split_name, cls)
            os.makedirs(cls_dest, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_src, img), os.path.join(cls_dest, img))

if __name__ == "__main__":
    src = "PlantVillage"  # change to "color" if that's your dataset folder
    dest = "data"
    print(f"Splitting from {src} -> {dest} (train/val/test)")
    split_folder(src, dest, val_frac=0.2, test_frac=0.1)
    print("Done.")
