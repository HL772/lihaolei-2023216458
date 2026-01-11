# 机器视觉实验

本仓库包含四个机器视觉课程的实验。

## 个人信息

- 姓名：李浩磊
- 学号：2023216458
- 班级：智科23-3班

- **实验一**: 图像滤波
- **实验二**: 车道线检测
- **实验三**: 手写数字识别
- **实验四**: 校园共享单车检测

---

## 实验一：图像处理

### 文件说明

- `实验一代码.ipynb`: 实验一的过程代码，包含了主要的图像处理和分析过程。
- `实验一报告.docx`: 实验一的详细报告。
- `input_image.jpg`: 原始输入图像。
- `loadnpy.py`: 一个Python脚本，用于加载生成的`.npy`格式文件`texture_features.npy`。
- `result/`: 存放实验结果的文件夹:
    - `texture_features.npy`: 保存了从图像中提取的纹理特征的NumPy数组文件。    
    - `result_custom_kernel.png`: 应用自定义卷积核后的结果图像。
    - `result_histogram.png`: 图像的灰度直方图。
    - `result_lbp_analysis.png`: LBP（局部二值模式）分析的结果图像。
    - `result_lbp_texture.png`: LBP纹理特征的可视化图像。
    - `result_sobel.png`: 应用Sobel边缘检测算子后的结果图像。

---

## 实验二：图像分类

### 文件说明

- `实验二代码.ipynb`: 实验二的实现，包含代码和各中间步骤说明与展示。
- `实验二报告.docx`: 实验二的详细报告。
- `campus_road.jpg`: 实验所使用的原始图片文件。
- `output_campus_road.jpg`: 经过实验处理后生成的图片文件。

---

## 实验三：手写数字识别

### 文件说明

- `实验三代码.ipynb`: 本实验的主要代码文件，它包含了模型训练、测试以及调用模型进行预测的完整代码。
- `实验三报告.docx`: 这是实验三的报告，详细记录了实验的步骤、结果和分析。
- `mnist_cnn_best.pth`: 这是`实验三代码.ipynb`训练的卷积神经网络（CNN）模型权重文件。可以用 `predict_id.py` 加载这个权重来进行预测。
- `predict_id.py`: 这是一个独立的Python脚本，用于加载 `mnist_cnn_best.pth` 模型来对 `my_id.jpg` 图片进行预测。
- `my_id.jpg`: 一张手写学号2023216458的图片，作为 `predict_id.py` 脚本的输入。
- `result.jpg`: 本实验的预测结果图片，包含原始图片以及模型预测的数字结果。

---

## 实验四：目标检测

### 文件说明

-   `实验四代码.ipynb`: 模型训练与可视化展示，包含实验的主要代码和步骤。
-   `实验四报告.docx`: 实验报告文档。
-   `detect_gui.py`: 图形化检测界面，可以加载训练出来的模型对图片、视频或摄像头进行实时目标检测。
-   `label_yolo_single.py`: 原始数据集处理，用于对单张图片进行YOLO格式的手动标注。
-   `prepare_yolo_dataset.py`: 准备YOLO微调训练数据集的脚本，用于划分训练集和验证集。
-   `train_result/`: 存放YOLOv8模型训练结果的目录。
    -   `weights/best.pt`: 训练过程中验证集上表现最好的模型权重。
-   `yolo_dataset/`: YOLO格式的数据集。
    -   `images/`: 存放原始图片。
    -   `labels/`: 存放标注文件。
    -   `data.yaml`: 数据集配置文件。
-   `test/`: 用于测试的图片目录。
