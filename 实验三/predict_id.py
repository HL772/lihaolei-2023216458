"""
脚本说明：加载实验三代码.ipynb训练的 MNIST CNN，对学号图片进行分割与逐位识别。
使用示例：python predict_id.py --weights mnist_cnn_best.pth  # 默认 my_id.jpg
核心可调：--image --invert --min-area --pad --min-width --device
"""
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# --------- 模型定义（与 notebook 相同） ---------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------- 学号图片分割（OpenCV 流程，与 notebook/predict_id 一致） ---------
def segment_id_image(
    path: str,
    invert: bool = True,
    min_area: int = 60,
    pad: int = 6,
    min_width: int = 8,
) -> List[Tuple[Image.Image, torch.Tensor]]:
    """
    OpenCV 分割流程：灰度+高斯模糊→Otsu 二值/反色→闭操作→Canny 行列投影粗裁剪→轮廓外接矩形分割；
    若轮廓不足则列投影切分连体。每段裁空白、方形填充、缩放到 28×28，并按 MNIST 均值方差归一化。
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, thresh_mode + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    edges = cv2.Canny(blur, 100, 200)

    rows_on = np.where(edges.sum(axis=1) > 0)[0]
    cols_on = np.where(edges.sum(axis=0) > 0)[0]
    if rows_on.size and cols_on.size:
        y0, y1 = rows_on[[0, -1]]
        x0, x1 = cols_on[[0, -1]]
        binary = binary[
            max(0, y0 - 5) : min(img.shape[0], y1 + 5),
            max(0, x0 - 5) : min(img.shape[1], x1 + 5),
        ]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area or h < 8:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])

    def split_by_projection(mask: np.ndarray) -> List[Tuple[int, int]]:
        col = (mask.sum(axis=0) > 0).astype(np.uint8)
        segs, s = [], None
        for i, v in enumerate(col):
            if v and s is None:
                s = i
            if not v and s is not None:
                if i - s >= min_width:
                    segs.append((s, i))
                s = None
        if s is not None and len(col) - s >= min_width:
            segs.append((s, len(col)))
        return segs

    norm = transforms.Normalize((0.1307,), (0.3081,))
    processed: List[Tuple[Image.Image, torch.Tensor]] = []
    if boxes:
        segments = [(x, y, x + w, y + h) for x, y, w, h in boxes]
    else:
        segments = [(l, 0, r, binary.shape[0]) for l, r in split_by_projection(binary)]

    for x0, y0, x1, y1 in segments:
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(binary.shape[1], x1 + pad)
        y1 = min(binary.shape[0], y1 + pad)
        digit = binary[y0:y1, x0:x1]
        if digit.sum() == 0:
            continue
        rows = digit.sum(axis=1) > 0
        cols = digit.sum(axis=0) > 0
        digit = digit[rows][:, cols]
        h, w = digit.shape
        side = max(h, w) + 8
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = digit
        pil = Image.fromarray(canvas)
        pil = pil.resize((28, 28), Image.BILINEAR)
        tensor = transforms.ToTensor()(pil)
        tensor = norm(tensor)
        processed.append((pil, tensor))
    return processed


# --------- 推理流程 ---------
def predict_digits(
    model: nn.Module, digits: List[Tuple[Image.Image, torch.Tensor]], device: torch.device
) -> str:
    model.eval()
    preds = []
    with torch.no_grad():
        for _, tensor in digits:
            logit = model(tensor.unsqueeze(0).to(device))
            preds.append(str(logit.argmax(dim=1).item()))
    return "".join(preds)


def main():
    parser = argparse.ArgumentParser(description="识别学号图片数字的命令行工具")
    parser.add_argument(
        "--image",
        default="my_id.jpg",
        help="Path to ID image (default: my_id.jpg).",
    )
    parser.add_argument(
        "--weights", default="mnist_cnn_best.pth", help="Path to trained weight file."
    )
    parser.add_argument("--invert", action="store_true", help="是否反色（二值为白字黑底，默认反色以适配 MNIST）。")
    parser.add_argument("--no-invert", dest="invert", action="store_false")
    parser.set_defaults(invert=True)
    parser.add_argument("--min-area", type=int, default=60, help="保留的最小轮廓面积。")
    parser.add_argument("--pad", type=int, default=6, help="单字裁剪周围的额外留白像素。")
    parser.add_argument(
        "--min-width", type=int, default=8, help="列投影切分时允许的最小宽度。"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="选择 cpu 或 cuda"
    )
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = torch.device(args.device)
    model = MNISTCNN().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    digits = segment_id_image(
        args.image,
        invert=args.invert,
        min_area=args.min_area,
        pad=args.pad,
        min_width=args.min_width,
    )
    if not digits:
        print("未检测到数字，尝试调整 --invert/--min-area/--pad/--min-width。")
        return
    id_number = predict_digits(model, digits, device)
    print(f"识别学号：{id_number}")
    print(f"分割位数：{len(digits)}")


if __name__ == "__main__":
    main()
