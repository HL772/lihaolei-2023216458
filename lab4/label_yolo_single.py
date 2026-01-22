#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise

import tkinter as tk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_index(stem: str) -> Optional[tuple[str, int]]:
    if "_" not in stem:
        return None
    prefix, num_str = stem.rsplit("_", 1)
    if not num_str.isdigit():
        return None
    return prefix, int(num_str)


def make_label_key(label_path: Path, dataset_dir: Path) -> str:
    try:
        return label_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return label_path.as_posix()


def load_done_set(done_file: Path) -> set[str]:
    if not done_file.exists():
        return set()
    lines = done_file.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def write_done_set(done_file: Path, done_set: set[str]) -> None:
    if not done_set:
        if done_file.exists():
            done_file.unlink()
        return
    content = "\n".join(sorted(done_set)) + "\n"
    done_file.write_text(content, encoding="utf-8")


def collect_items(dataset_dir: Path) -> list[tuple[Path, Path, str]]:
    items: list[tuple[Path, Path, str]] = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                label_path = lbl_dir / f"{img_path.stem}.txt"
                items.append((img_path, label_path, split))

    def sort_key(item: tuple[Path, Path, str]):
        img_path, _, split = item
        parsed = parse_index(img_path.stem)
        if parsed is None:
            return (1, img_path.name.lower())
        prefix, idx = parsed
        return (0, prefix, idx, split, img_path.name.lower())

    items.sort(key=sort_key)
    return items


def is_labeled(
    label_path: Path,
    done_set: Optional[set[str]] = None,
    dataset_dir: Optional[Path] = None,
) -> bool:
    if label_path.exists() and label_path.stat().st_size > 0:
        return True
    if done_set is None or dataset_dir is None:
        return False
    key = make_label_key(label_path, dataset_dir)
    return key in done_set


class Labeler:
    def __init__(self, items, class_id: int, redo: bool, dataset_dir: Path):
        self.items = items
        self.class_id = class_id
        self.redo = redo
        self.index = 0
        self.dataset_dir = dataset_dir
        self.done_file = dataset_dir / ".labeler_done"
        self.done_set = load_done_set(self.done_file)

        self.root = tk.Tk()
        self.root.title("YOLO Single Object Labeler")
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.canvas = tk.Canvas(self.root, cursor="tcross")
        self.canvas.pack()

        self.photo = None
        self.image = None
        self.orig_w = 0
        self.orig_h = 0
        self.scale = 1.0
        self.rect_id = None
        self.start_xy = None
        self.box_xy = None
        self.current_label = None
        self.current_image = None
        self.current_split = None

        self.root.bind("<Key>", self.on_key)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.load_next()

    def on_quit(self):
        self.root.quit()

    def load_next(self):
        while self.index < len(self.items):
            img_path, label_path, split = self.items[self.index]
            if not self.redo and is_labeled(label_path, self.done_set, self.dataset_dir):
                self.index += 1
                continue
            self.current_image = img_path
            self.current_label = label_path
            self.current_split = split
            self.load_image(img_path)
            return
        print("All images labeled.")
        self.root.quit()

    def load_image(self, img_path: Path):
        self.image = Image.open(img_path)
        self.orig_w, self.orig_h = self.image.size

        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.9)
        self.scale = min(1.0, max_w / self.orig_w, max_h / self.orig_h)

        if self.scale < 1.0:
            disp_w = max(1, int(self.orig_w * self.scale))
            disp_h = max(1, int(self.orig_h * self.scale))
            display_img = self.image.resize((disp_w, disp_h), Image.LANCZOS)
        else:
            disp_w = self.orig_w
            disp_h = self.orig_h
            display_img = self.image

        self.photo = ImageTk.PhotoImage(display_img)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.rect_id = None
        self.start_xy = None
        self.box_xy = None

        total = len(self.items)
        title = f"[{self.index + 1}/{total}] {img_path.name} ({self.current_split})"
        self.root.title(title)
        print(f"Labeling {img_path}")

    def clamp_xy(self, x, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return x, y

    def on_press(self, event):
        x, y = self.clamp_xy(event.x, event.y)
        self.start_xy = (x, y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.box_xy = None

    def on_drag(self, event):
        if self.start_xy is None:
            return
        x0, y0 = self.start_xy
        x1, y1 = self.clamp_xy(event.x, event.y)
        if self.rect_id is None:
            self.rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="red", width=2
            )
        else:
            self.canvas.coords(self.rect_id, x0, y0, x1, y1)

    def on_release(self, event):
        if self.start_xy is None:
            return
        x0, y0 = self.start_xy
        x1, y1 = self.clamp_xy(event.x, event.y)
        self.start_xy = None
        self.box_xy = (x0, y0, x1, y1)

    def on_key(self, event):
        key = event.keysym.lower()
        if key in {"s", "return"}:
            self.save_and_next()
        elif key in {"n", "0"}:
            self.mark_no_object()
        elif key == "r":
            self.reset_box()
        elif key in {"q", "escape"}:
            self.on_quit()

    def reset_box(self):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = None
        self.box_xy = None

    def save_and_next(self):
        if self.box_xy is None:
            print("No box selected. Drag to draw a box before saving.")
            return

        x0, y0, x1, y1 = self.box_xy
        x_min = min(x0, x1) / self.scale
        y_min = min(y0, y1) / self.scale
        x_max = max(x0, x1) / self.scale
        y_max = max(y0, y1) / self.scale

        x_min = max(0.0, min(x_min, self.orig_w - 1))
        y_min = max(0.0, min(y_min, self.orig_h - 1))
        x_max = max(0.0, min(x_max, self.orig_w - 1))
        y_max = max(0.0, min(y_max, self.orig_h - 1))

        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w <= 1 or box_h <= 1:
            print("Box too small. Please draw a larger box.")
            return

        x_center = (x_min + x_max) / 2.0 / self.orig_w
        y_center = (y_min + y_max) / 2.0 / self.orig_h
        w_norm = box_w / self.orig_w
        h_norm = box_h / self.orig_h

        label_line = (
            f"{self.class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        )
        self.current_label.write_text(label_line, encoding="utf-8")
        self.set_done(self.current_label, False)
        print(f"Saved: {self.current_label}")

        self.index += 1
        self.load_next()

    def mark_no_object(self):
        if self.current_label is None:
            return
        self.current_label.write_text("", encoding="utf-8")
        self.set_done(self.current_label, True)
        print(f"Marked empty: {self.current_label}")

        self.index += 1
        self.load_next()

    def set_done(self, label_path: Path, done: bool):
        key = make_label_key(label_path, self.dataset_dir)
        changed = False
        if done:
            if key not in self.done_set:
                self.done_set.add(key)
                changed = True
        else:
            if key in self.done_set:
                self.done_set.remove(key)
                changed = True
        if changed:
            write_done_set(self.done_file, self.done_set)

    def run(self):
        print(
            "Controls: drag to draw; S/Enter to save; N/0 = no object; "
            "R to reset; Q to quit."
        )
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label a single object per image for a YOLO dataset."
    )
    parser.add_argument(
        "--dataset",
        default="yolo_dataset",
        help="Dataset root with images/ and labels/ (default: yolo_dataset).",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class id to write into labels (default: 0).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Relabel even if a label file already exists.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    items = collect_items(dataset_dir)
    if not items:
        print("No images found under images/train or images/val.", file=sys.stderr)
        return 1

    labeler = Labeler(items, class_id=args.class_id, redo=args.redo, dataset_dir=dataset_dir)
    labeler.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
