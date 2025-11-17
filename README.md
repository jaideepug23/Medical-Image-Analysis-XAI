# Explainable AI for Medical Image Analysis

## A Simple Approach Using Semantic Segmentation and Attention Mapping

### Authors
- **Jaideep** (jaideep.ug23@nsut.ac.in)
- **Rishabh Singh** (rishabh.singh.ug23@nsut.ac.in)
- **Chinmay Solanki** (chinmay.solanki.ug23@nsut.ac.in)

**Department of Computer Science and Engineering**  
**Netaji Subhas University of Technology (NSUT)**

---

## 📋 Abstract

Artificial Intelligence (AI) and Deep Learning have become very popular in the medical field. They help in detecting diseases and identifying affected areas in medical images. However, most AI models work like a "black box" - they give answers but do not explain how they reached those answers.

In medical science, it is very important to understand the reason behind every decision because it affects patient health. This paper presents a simple method of **Explainable AI (XAI)** that combines **semantic segmentation** and **attention mapping**. This approach helps to show clearly which parts of an image influenced the model's output, making the results more understandable and trustworthy for doctors.

---

## 🎯 Key Features

- ✅ **Semantic Segmentation** using U-Net architecture
- ✅ **Attention Mapping** for visual explanations
- ✅ **High Accuracy**: Training Accuracy 84%, Validation Accuracy 82%
- ✅ **Dice Coefficient**: 0.87
- ✅ **IoU Score**: 0.81
- ✅ **Explainable Results** for medical professionals

---

## 📊 Dataset

### ISIC 2018 - Skin Lesion Analysis Dataset

This project uses the **ISIC 2018 dataset** (Codella et al., 2019) which contains thousands of skin lesion images with corresponding segmentation masks that show the exact boundary of each lesion.

**Dataset Features:**
- Multiple skin lesion images from different patients
- Variations in color, size, and lighting conditions
- Ground truth segmentation masks
- Suitable for testing explainability methods

**Download Dataset:**
- Google Drive: [HAM10000 Dataset](https://drive.google.com/file/d/1are4o0A9Y6ZcGuOOb30i34MuOJZKW7CC/view?usp=drivesdk)
- Official Source: [ISIC Archive](https://challenge.isic-archive.com/)

---

## 🏗️ Methodology

### 1. Segmentation Model (U-Net)

The U-Net model was used for semantic segmentation with:
- **Encoder**: Extracts important features
- **Decoder**: Reconstructs the segmented image
- **Loss Function**: Combination of Dice Loss and Binary Cross-Entropy Loss

### 2. Attention Mapping

Attention maps were created using a CAM-based method to show which parts of the image were most important for the prediction.

### 3. Combined Approach

By overlaying attention maps with segmentation masks, we create visual explanations that help doctors understand AI decisions.

---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | 84% |
| Validation Accuracy | 82% |
| Dice Coefficient | 0.87 |
| IoU (Intersection over Union) | 0.81 |
| Final Loss | 0.29 |

### Comparison with Existing Methods

| Method | Accuracy | Explainability Level |
|--------|----------|---------------------|
| Existing Model 1 | 76% | Low |
| U-Net + Grad-CAM | 82% | Medium |
| **Proposed Model** | **84%** | **High** |

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Required Libraries
```bash
pip install torch torchvision
pip install opencv-python-headless
pip install Pillow numpy matplotlib
pip install scikit-image scikit-learn seaborn
pip install basicsr facexlib gfpgan realesrgan
pip install pydicom nibabel medpy
```

---

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/jaideepug23/Medical-Image-Analysis-XAI.git
cd Medical-Image-Analysis-XAI
```

### 2. Download Dataset
Download the HAM10000 dataset from the link provided above and extract it to the `datasets/` folder.

### 3. Run the Notebook
```bash
jupyter notebook Untitled0.ipynb
```

Or upload to Google Colab for GPU acceleration.

---

## 📁 Project Structure

```
Medical-Image-Analysis-XAI/
│
├── Untitled0.ipynb                 # Main Jupyter notebook
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── research_paper.pdf              # Detailed research paper
│
├── datasets/
│   └── ham10000/                   # Dataset directory
│       ├── HAM10000_images_part1/
│       ├── HAM10000_images_part2/
│       └── HAM10000_metadata.csv
│
└── results/
    ├── segmentation_outputs/       # Segmentation results
    └── attention_maps/             # Generated attention maps
```

---

## 🔬 Technical Details

### Model Architecture
- **Input Size**: 224 × 224 × 3
- **Convolutional Layers**: 4 encoder blocks, 4 decoder blocks
- **Activation**: ReLU
- **Pooling**: MaxPooling (2×2)
- **Dropout**: 0.5
- **Output Classes**: Varies based on dataset

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16
- **Epochs**: 10
- **Device**: CUDA (if available)

---

## 🎓 Research Paper

For detailed methodology, experiments, and results, please refer to the full research paper included in this repository.

**Citation:**
```bibtex
@article{jaideep2025explainable,
  title={Explainable AI for Medical Image Analysis using Semantic Segmentation and Attention Mapping},
  author={Jaideep and Singh, Rishabh and Solanki, Chinmay},
  journal={Department of Computer Science and Engineering, NSUT},
  year={2025}
}
```

---

## 🔑 Key Insights

1. **Transparency**: Attention maps show which regions influenced predictions
2. **Trust**: Visual explanations help doctors validate AI decisions
3. **Error Detection**: Misaligned attention reveals model errors
4. **Clinical Applicability**: Results are interpretable by medical professionals

---

## 🔮 Future Work

- [ ] Extend to other medical imaging modalities (CT, MRI)
- [ ] Implement transformer-based attention mechanisms
- [ ] Integrate human feedback for model improvement
- [ ] Deploy in hospital diagnostic systems
- [ ] Multi-disease classification support

---

## 📚 References

1. Ribeiro et al. (2016) - "Why Should I Trust You?" - KDD Conference
2. Ronneberger et al. (2015) - U-Net: Convolutional Networks for Biomedical Image Segmentation
3. Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
4. Codella et al. (2019) - Skin Lesion Analysis Toward Melanoma Detection 2018
5. Doshi-Velez & Kim (2017) - Towards a Rigorous Science of Interpretable Machine Learning

---

## 📧 Contact

For questions, collaborations, or feedback:

- **Jaideep**: jaideep.ug23@nsut.ac.in
- **Rishabh Singh**: rishabh.singh.ug23@nsut.ac.in
- **Chinmay Solanki**: chinmay.solanki.ug23@nsut.ac.in

**Institution**: Netaji Subhas University of Technology (NSUT), Delhi

---

## 📝 License

This project is open-source and available for academic and research purposes.

---

## ⭐ Acknowledgments

We thank the faculty of the Department of Computer Science and Engineering, Netaji Subhas University of Technology, for their continuous support and guidance during this research. Special gratitude to **Dr. Ankush Jain** and **Prof. Rohit Kumar Ahlawat** for their invaluable mentorship and guidance.

Special thanks to the ISIC Archive for providing the dataset used in this study.

---

**Made with ❤️ by NSUT Students | 2025**
