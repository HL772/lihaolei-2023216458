# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

try:
    from PIL import Image, ImageTk
except Exception as exc: 
    raise SystemExit(
        "Pillow is required for the GUI. Install with: pip install pillow"
    ) from exc

from ultralytics import YOLO


class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LHL的共享单车检测器")
        self.root.configure(bg="#f5f7fb")

        self.model = None
        self.image_path = None

        # --- 修改开始 ---
        # 1. 获取当前脚本文件所在的绝对目录
        base_dir = Path(__file__).resolve().parent
        
        # 2. 使用 base_dir 拼接权重文件的路径
        # 这样无论你在哪里运行脚本，它都会去脚本同级目录下的 train_result 里面找
        default_weights = base_dir / "train_result" / "weights" / "best.pt"
        
        if not default_weights.exists():
            # 备选方案：检查当前目录下是否有 yolov8n.pt
            if Path("yolov8n.pt").exists():
                default_weights = Path("yolov8n.pt")
        # --- 修改结束 ---

        self.weights_var = tk.StringVar(value=str(default_weights))
        self.image_var = tk.StringVar(value="")
        self.conf_var = tk.IntVar(value=70)

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#f5f7fb",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#f5f7fb",
            foreground="#64748b",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Hint.TLabel",
            background="#ffffff",
            foreground="#64748b",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "TButton",
            font=("Microsoft YaHei UI", 10),
            padding=(10, 6),
        )
        style.configure(
            "Accent.TButton",
            background="#2563eb",
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8")],
            foreground=[("active", "#ffffff")],
        )
        style.configure(
            "Section.TLabelframe",
            background="#ffffff",
        )
        style.configure(
            "Section.TLabelframe.Label",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#f5f7fb",
            foreground="#475569",
            font=("Microsoft YaHei UI", 9),
        )

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=(16, 12))
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="LHL的共享单车检测器", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header, text="选择图片并查看检测结果", style="Subtitle.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        ttk.Separator(outer).pack(fill=tk.X, pady=(0, 12))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        model_frame = ttk.LabelFrame(
            left, text="模型设置", style="Section.TLabelframe", padding=(12, 8)
        )
        model_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(model_frame, text="权重文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(model_frame, textvariable=self.weights_var).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 6)
        )
        btn_row = ttk.Frame(model_frame, style="Card.TFrame")
        btn_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(btn_row, text="浏览", command=self._browse_weights).pack(
            side=tk.LEFT
        )
        ttk.Button(
            btn_row, text="加载模型", style="Accent.TButton", command=self._load_model
        ).pack(side=tk.LEFT, padx=(6, 0))
        model_frame.columnconfigure(0, weight=1)

        image_frame = ttk.LabelFrame(
            left, text="图片设置", style="Section.TLabelframe", padding=(12, 8)
        )
        image_frame.pack(fill=tk.X)
        ttk.Label(image_frame, text="图片文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(image_frame, textvariable=self.image_var).grid(
            row=1, column=0, sticky="ew", pady=(4, 6)
        )
        ttk.Button(image_frame, text="浏览", command=self._browse_image).grid(
            row=1, column=1, padx=(6, 0)
        )
        conf_row = ttk.Frame(image_frame, style="Card.TFrame")
        conf_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(conf_row, text="置信度(%)").pack(side=tk.LEFT)
        ttk.Label(conf_row, textvariable=self.conf_var, style="Hint.TLabel").pack(
            side=tk.RIGHT
        )
        self.conf_scale = tk.Scale(
            image_frame,
            variable=self.conf_var,
            from_=30,
            to=95,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=220,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            troughcolor="#e2e8f0",
        )
        self.conf_scale.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        ttk.Button(
            image_frame, text="开始检测", style="Accent.TButton", command=self._detect
        ).grid(row=4, column=0, columnspan=2, sticky="ew")
        image_frame.columnconfigure(0, weight=1)

        preview = ttk.Frame(right)
        preview.pack(fill=tk.BOTH, expand=True)
        self.orig_label = tk.Label(
            preview,
            text="原图",
            bg="#f8fafc",
            fg="#94a3b8",
            font=("Microsoft YaHei UI", 12),
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        self.pred_label = tk.Label(
            preview,
            text="预测结果",
            bg="#f8fafc",
            fg="#94a3b8",
            font=("Microsoft YaHei UI", 12),
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        self.orig_label.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.pred_label.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        preview.columnconfigure(0, weight=1, minsize=280)
        preview.columnconfigure(1, weight=2, minsize=520)
        preview.rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor="w", pady=(10, 0)
        )

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _browse_weights(self):
        path = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)
            self.image_path = path
            self._show_image(path, self.orig_label)

    def _load_model(self):
        weights = self.weights_var.get().strip()
        if not weights or not Path(weights).exists():
            messagebox.showerror("错误", "未找到权重文件。")
            return
        self._set_status("正在加载模型...")
        try:
            self.model = YOLO(weights)
        except Exception as exc:
            messagebox.showerror("错误", f"模型加载失败: {exc}")
            self._set_status("模型加载失败")
            return
        self._set_status("模型已加载")

    def _show_image(self, path_or_array, target_label, max_size=(520, 360)):
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
            if img is None:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = path_or_array

        self.root.update_idletasks()
        max_w = max(200, target_label.winfo_width() - 20)
        max_h = max(200, target_label.winfo_height() - 20)
        max_size = (max_w, max_h)
        pil_img = Image.fromarray(img)
        pil_img.thumbnail(max_size)
        photo = ImageTk.PhotoImage(pil_img)
        target_label.configure(image=photo, text="")
        target_label.image = photo

    def _detect(self):
        if not self.image_var.get():
            messagebox.showwarning("提示", "请先选择图片。")
            return
        if self.model is None:
            self._load_model()
            if self.model is None:
                return

        conf = float(self.conf_var.get()) / 100.0
        self._set_status("正在检测...")
        try:
            results = self.model.predict(
                source=self.image_var.get(), conf=conf, verbose=False
            )
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self._show_image(annotated, self.pred_label)
            num_det = len(results[0].boxes) if hasattr(results[0], "boxes") else 0
            self._set_status(f"检测完成：{num_det} 个目标")
        except Exception as exc:
            messagebox.showerror("错误", f"检测失败: {exc}")
            self._set_status("检测失败")


def main():
    root = tk.Tk()
    app = DetectorApp(root)
    root.geometry("1160x700")
    root.minsize(980, 620)
    root.mainloop()


if __name__ == "__main__":
    main()