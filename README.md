# 🔍 Deepfake Detection with CNN Models

A deep learning-based project comparing the performance of ResNet50, MobileNetV2, and EfficientNetV2B0 architectures in detecting deepfake videos using frame-level and video-level evaluation metrics.

## 📘 Project Overview

Deepfakes—AI-generated synthetic videos—pose a serious threat to online trust, misinformation, and privacy. This project evaluates three popular Convolutional Neural Network (CNN) architectures for deepfake detection:

- **ResNet50**: Deep architecture with residual connections.
- **MobileNetV2**: Lightweight and optimized for mobile deployment.
- **EfficientNetV2B0**: Scaled architecture balancing accuracy and efficiency.

---

## 📁 Dataset

We used the **Deep Fake Detection (DFD)** dataset from Kaggle:
- Videos labeled as either `real` or `fake`.
- 508 videos for training (254 real, 254 fake).
- 220 videos for testing (110 real, 110 fake).
- 10 equally spaced frames extracted per video.
- Augmentations applied: flipping, rotation, brightness adjustments.

---

## 🧠 Models & Methodologies

### ✅ ResNet50 Pipeline
- Pretrained on ImageNet, with the final layers fine-tuned.
- Variants trained using different input sizes (128x128, 64x64, 32x32) and learning rates.
- Models experimented with dropout, batch normalization, L2 regularization.
- Best accuracy achieved: **92.27% video-level** with `128x128` input at `5e-5` learning rate.

### ✅ MobileNetV2 Pipeline
- Best suited for real-time applications.
- Fast training with reduced complexity.
- Best accuracy achieved: **90.91% video-level** with `128x128` input at `1e-4` learning rate.

### ✅ EfficientNetV2B0 Pipeline
- Fine-tuned with moderate, light, and heavy regularization variants.
- Frame-level and video-level evaluation.
- Lower overall performance than ResNet50 and MobileNetV2 in this setup, maxing around **61.02%** video-level accuracy.

---

## 🧪 Evaluation Metrics

- **Accuracy** (Frame-Level & Video-Level)
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**
- **Loss Curves and Training Graphs**

---

## 💻 Technologies Used

- Python 3.8+
- PyTorch / TensorFlow 2.x
- OpenCV
- FFmpeg
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Kaggle & Google Colab (GPU-enabled)

---

## ⚙️ How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/deepfake-detection-cnn.git
    cd deepfake-detection-cnn
    ```

2. Setup environment:
    ```bash
    pip install -r requirements.txt
    ```

3. Organize dataset:
    ```
    dataset/
    ├── real/
    └── fake/
    ```

4. Run training:
    ```bash
    python train_resnet.py
    python train_mobilenet.py
    python train_efficientnet.py
    ```

5. Evaluate:
    ```bash
    python evaluate.py --model resnet50 --input-size 128
    ```

---

## 📊 Results Summary

| Model         | Input Size | Learning Rate | Video Acc. | Frame Acc. |
|---------------|------------|----------------|-------------|-------------|
| ResNet50      | 128x128    | 5e-5           | 92.27%      | 91.31%      |
| MobileNetV2   | 128x128    | 1e-4           | 90.91%      | 88.95%      |
| EfficientNetV2B0 | 128x128 | 5e-5           | 61.02%      | 59.66%      |

---

## 📚 References

See full reference list in the [Report](./Deepfake%20Detection%20Report.pdf).

---

## ✍️ Authors

- **Nikhilesh Rachabattula** - [@nikhilesh](https://github.com/Nikhilesh-rachabattula)
- **Kanuri Sai Kiran** - [@saikiran](https://github.com/saikiran-kanuri)
- **P. Dwijesh Reddy** - [@dwijesh](https://github.com/Dwijesh05)

---

## 🧾 License

This project is for academic and research use only.

---

## 📎 Acknowledgements

- Dr. E. Sreenivasa Reddy (Project Guide)
- VIT-AP University
- Kaggle for GPU support and dataset hosting

