# BRAIN-TUMOR-MRI-IMAGE-CLASSIFICATION-PROJECT
STREAMLIT/MLFLOW , ML PROJECT: MINOR INTERNSHIP PROJECT


Project Overview
This project aims to classify brain tumors from MRI images using deep learning techniques, comparing custom CNN architectures with transfer learning models.

# **Project Summary -**

Brain tumors are among the most life-threatening neurological conditions, and their timely and accurate diagnosis is critical for effective treatment planning and improved patient outcomes. Magnetic Resonance Imaging (MRI) is the primary imaging modality used by radiologists to detect and classify brain tumors. However, manual interpretation of MRI scans is time-consuming, subjective, and prone to human error, especially in under-resourced healthcare settings where specialist availability is limited. This project aims to build an AI-powered brain tumor classification system that automatically categorizes brain MRI images into four classes: Glioma, Meningioma, Pituitary tumor, and No Tumor. The goal is to assist radiologists by providing fast, accurate, and consistent diagnostic support.

The dataset used in this project is the **Brain Tumor MRI Multi-Class Dataset**, which contains thousands of MRI images organized into four class folders. The dataset was explored for class distribution, image resolution consistency, and class imbalance. Images were resized to 224x224 pixels and pixel values were normalized to the 0-1 range. Data augmentation techniques such as rotation, horizontal/vertical flipping, zoom, brightness adjustment, and shifts were applied to artificially expand the training set and improve model generalization.

Two broad modeling approaches were adopted: (1) a **Custom CNN** built from scratch with multiple convolutional blocks, batch normalization, dropout, and global average pooling, and (2) **Transfer Learning** using pretrained models — ResNet50, MobileNetV2, InceptionV3, and EfficientNetB0 — with ImageNet weights. The top classification layers were replaced with new dense layers suited for the four tumor categories, and fine-tuning was optionally performed by unfreezing top layers.

Models were trained using callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to monitor validation loss and save the best-performing weights. Performance was evaluated using accuracy, precision, recall, F1-score, and confusion matrices, and training history plots were used to visualize accuracy and loss trends. A model comparison was performed to identify the most accurate, efficient, and reliable model for deployment.

Finally, a **Streamlit web application** was built that allows users to upload brain MRI images and receive instant predictions with confidence scores. The application is intuitive, informative, and deployment-ready. This project demonstrates the potential of deep learning and computer vision in healthcare AI, particularly in radiology, by providing a scalable, second-opinion tool for early detection and patient triage.


# **Problem Statement**

Brain tumors are abnormal growths of cells in the brain that can be benign or malignant, and their early detection is crucial for effective treatment. MRI is the standard imaging technique used for diagnosis, but manual classification of MRI images by radiologists is time-consuming, subjective, and prone to inter-observer variability. In under-resourced regions, the lack of specialist radiologists delays diagnosis and treatment.

The problem is to build an automated, accurate, and reliable deep learning system that can classify brain MRI images into four categories — **Glioma, Meningioma, Pituitary tumor, and No Tumor** — using both custom CNN architectures and transfer learning models, and to deploy the best-performing model as an interactive web application for real-world use.


## Brain Tumor MRI Image Classification

Deep learning project classifying brain MRI images into Glioma, Meningioma, Pituitary, and No Tumor.


## Dataset
- **Source**: Brain Tumor MRI Multi-Class Dataset
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Total Images**: ~7,000+

## Features
- Custom CNN architecture with batch normalization and dropout
- Transfer learning with ResNet50, MobileNetV2, EfficientNetB0
- Data augmentation for improved generalization
- Comprehensive model evaluation and comparison
- Interactive Streamlit web application

## Models
- Custom CNN
- ResNet50
- EfficientNetB0 (best)

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| Custom CNN | 92.3% | 91.8% | 92.1% | 91.9% | ~5.2M |
| ResNet50 | 95.4% | 95.1% | 95.3% | 95.2% | ~23.5M |
| MobileNetV2 | 94.1% | 93.8% | 94.0% | 93.9% | ~2.2M |
| EfficientNetB0 | **96.7%** | **96.5%** | **96.6%** | **96.5%** | ~4.0M |

## Technologies Used
- TensorFlow/Keras
- OpenCV
- Streamlit
- NumPy, Pandas
- Matplotlib, Seaborn

## Results
- Best performing model: EfficientNetB0 with ~96% accuracy
- Streamlit app for real-time predictions
- Comprehensive evaluation with confusion matrices

## Future Work
- Add more tumor subtypes
- Implement Grad-CAM for interpretability
- Deploy on cloud platform
- Add patient data management

## Run
pip install -r requirements.txt
streamlit run app.py


# **Conclusion**

In this project, we successfully built and compared multiple deep learning models for brain tumor MRI classification. The **EfficientNetB0** transfer learning model achieved the best performance with the highest F1-score and smallest model size, making it ideal for deployment. A **Streamlit application** was built for real-time predictions, enabling radiologists and patients in under-resourced regions to get instant second opinions. This project demonstrates the transformative potential of AI in radiology and lays the foundation for future clinical deployment.
