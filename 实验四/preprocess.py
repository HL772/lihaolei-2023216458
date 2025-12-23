"""
对 photo 文件夹中的共享单车照片做预处理（重命名、等比例缩放、划分训练/验证），
输出到 dataset/images/train 与 dataset/images/val，生成清单 manifest.json。
"""

from pathlib import Path
from PIL import Image
import json
import random

# 源目录与输出目录
SRC_DIR = Path("photo")
OUT_DIR = Path("dataset")

# 预处理参数
MAX_DIM = 1280       # 最长边缩放到不超过该值（保持长宽比）
TRAIN_RATIO = 0.8    # 训练集占比，其余为验证集
SEED = 42            # 固定随机种子，保证划分可复现

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images():
    """收集源图片路径列表。"""
    assert SRC_DIR.exists(), f"未找到源目录: {SRC_DIR}"
    files = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise ValueError(f"{SRC_DIR} 中未找到可处理的图片。")
    files.sort()
    return files


def process_image(src_path: Path, dst_path: Path):
    """读取->RGB->等比例缩放->保存为 JPEG，返回元信息。"""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, format="JPEG", quality=90)

    return {
        "orig_file": str(src_path),
        "file": str(dst_path),
        "width": img.width,
        "height": img.height,
        "scale": scale,
    }


def split_dataset(files):
    """按照 TRAIN_RATIO 划分训练/验证列表。"""
    rnd = random.Random(SEED)
    rnd.shuffle(files)
    split_idx = max(1, int(len(files) * TRAIN_RATIO))  # 至少保证训练集中有 1 张
    return files[:split_idx], files[split_idx:]


def main():
    files = collect_images()
    train_files, val_files = split_dataset(files)

    manifest = {
        "max_dim": MAX_DIM,
        "train_ratio": TRAIN_RATIO,
        "seed": SEED,
        "images": [],
    }

    for split, subset in [("train", train_files), ("val", val_files)]:
        for idx, src in enumerate(subset, start=1):
            # 统一重命名，便于后续标注/训练
            dst_name = f"bike_{split}_{idx:03d}.jpg"
            dst_path = OUT_DIR / "images" / split / dst_name
            info = process_image(src, dst_path)
            info.update({"split": split, "name": dst_name})
            manifest["images"].append(info)
            print(f"[{split}] {src.name} -> {dst_path.name} | {info['width']}x{info['height']} scale={info['scale']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"预处理完成，清单已保存到 {manifest_path}")
    print(f"训练图片: {len(train_files)}，验证图片: {len(val_files)}")


if __name__ == "__main__":
    main()
