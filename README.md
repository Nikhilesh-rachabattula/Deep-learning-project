Deepfake Detection using CNN Models
Overview
This project focuses on evaluating and comparing the performance of various Convolutional Neural Network (CNN) models for deepfake video detection. With the rapid advancement of deepfake technology, the ability to accurately distinguish between genuine and manipulated content has become crucial for maintaining media integrity and combating misinformation. This study investigates ResNet50, MobileNetV2, and EfficientNetV2B0, analyzing their effectiveness in identifying deepfake videos based on various metrics.

Problem Statement
The proliferation of highly realistic deepfake videos poses significant challenges in areas such as journalism, law enforcement, and social media. Traditional detection methods are often insufficient against sophisticated manipulations. This research addresses the need for a comprehensive comparison of popular CNN models to determine which architecture offers the best trade-off between performance and efficiency for deepfake detection.

Objectives
To perform a comparative analysis of ResNet50, MobileNetV2, and EfficientNetV2B0 for deepfake video detection.

To evaluate the impact of different learning rates and image input sizes on model performance.

To assess models based on frame-level accuracy, video-level accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC score.

To explore the trade-offs between computational efficiency, training time, and model performance.

Methodology
Data Acquisition and Preparation
The project utilizes the Deep Fake Detection (DFD) dataset from Kaggle, comprising both real and fake videos.

Dataset Balancing: The number of fake videos is adjusted to match real videos to prevent bias.

Train-Test Split: A 70-30 train-test split is applied at the video level to ensure independent evaluation.

Training Set: 254 real and 254 fake videos (508 total)

Test Set: 110 real and 110 fake videos (220 total)

Frame Sampling: 10 uniformly spaced frames are extracted from each video.

Preprocessing: Frames are resized to standard resolutions (e.g., 128×128), normalized (pixel values scaled to [0,1]), and labeled.

Data Augmentation: Techniques like random horizontal flipping, rotation, zoom, and brightness adjustment are applied to the training set to enhance generalization.

Model Architectures
The study compares three CNN architectures, all fine-tuned from pre-trained ImageNet weights:

ResNet50-based Models:

Model V1 (Methodology 1): Uses a pre-trained ResNet50 as a frozen feature extractor with added layers (Global Average Pooling, Dropout, Dense sigmoid output). Trained with varying learning rates and image sizes (128×128, 64×64, 32×32).

Model V2 (Methodology 2): ResNet50 backbone with only the last 10 layers unfrozen for partial fine-tuning. Includes Global Average Pooling, Batch Normalization, Dropout, and Dense layers. Experiments conducted with different learning rates and dropout rates.

Model V3 (Methodology 3): Fully trainable ResNet50 base with varying regularization approaches (L2 regularization, different dropout rates). Three variants (V1, V2, V3) based on regularization strength.

EfficientNetV2-B0-based Models (Methodology 2):

Three architectural variants (V1, V2, V3) built upon the EfficientNetV2-B0 backbone, differing in dropout rates and L2 weight regularization. All variants use a frozen pre-trained backbone that is later unfrozen for fine-tuning.

MobileNetV2-based Models (Methodology 2):

Similar to EfficientNetV2-B0, three variants (V1, V2, V3) are explored with varying regularization and dropout, built on the MobileNetV2 backbone.

Training Strategy
Optimizer: Adam optimizer.

Loss Function: Binary cross-entropy.

Epochs: Up to 20 epochs, with early stopping if validation loss does not improve for 7 consecutive epochs.

Optimization: Mixed-precision computation and Just-In-Time (JIT) compilation enabled for improved GPU throughput.

Environment: Cloud-based platforms like Kaggle Notebooks and Google Colab (NVIDIA Tesla T4 GPUs, up to 29 GB RAM).

Evaluation Metrics
Model performance is evaluated at both frame-level and video-level using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC score

Results
Key Findings
EfficientNetV2B0 generally yielded the best results, demonstrating high accuracy and strong generalization across videos.

ResNet50 also performed well, offering a good balance between speed and accuracy.

MobileNetV2 was the fastest and most resource-efficient model, suitable for real-time or mobile applications, though with slightly lower accuracy compared to the other two.

Performance Highlights (Example from the report, specific values may vary based on configuration):
ResNet50 (Methodology 1, 128×128 frame size, Learning Rate 5e-5):

Frame Level Accuracy: 90.59%

Video Level Accuracy: 90.91%

Frame Level AUC: 0.92

Video Level AUC: 0.94

EfficientNetV2B0 (Methodology 2, 128×128 frame size, Learning Rate 1e-4):

Model_V2 (Light Regularization) showed the best performance among EfficientNet models in this configuration.

Frame Level Accuracy: 59.66%

Video Level Accuracy: 61.02%

Frame Level AUC: 0.62

Video Level AUC: 0.63

MobileNetV2 (Methodology 2, 128×128 frame size, Learning Rate 1e-4):

Model_V1 (Moderate Regularization) showed strong performance.

Frame Level Accuracy: 88.95%

Video Level Accuracy: 90.91%

Frame Level AUC: 0.92

Video Level AUC: 0.93

Hardware and Software Requirements
Hardware: NVIDIA Tesla T4 GPUs (Kaggle Notebooks, Google Colab)

Software:

Deep Learning Frameworks: PyTorch (preferred), TensorFlow

Libraries: OpenCV, Dlib, FFmpeg, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Future Work
Face Detection and Alignment: Incorporate face-centric preprocessing to improve sensitivity to local manipulations.

Temporal Modeling: Integrate dedicated temporal networks (e.g., TCNs) to capture motion inconsistencies more effectively.

Model Ensembling: Explore combining multiple trained variants for enhanced robustness.

Efficient I/O: Implement specialized video loaders and caching.

On-Device Deployment: Investigate techniques like quantization and pruning for real-time detection on mobile/embedded platforms.

Contributors
Nikhilesh Rachabattula (22BCE8410)

Kanuri Sai Kiran (22BCE7272)

P. Dwijesh Reddy (22BCE9104)

Supervisor
Dr. E. Sreenivasa Reddy, Professor-HAG, SCOPE, VIT-AP.

License
[Specify your license here, e.g., MIT, Apache 2.0, etc.]
