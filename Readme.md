# 🩺 Skin Lesion Risk Checker

An AI-powered **skin lesion risk-checking prototype** built with **PyTorch, EfficientNet-B0, and Streamlit**.

The application takes an uploaded skin-lesion image and predicts which of **7 lesion categories** it most closely resembles. It also calculates a combined risk score for potentially concerning categories and uses **Grad-CAM** to show which regions of the image influenced the model's prediction.

> ⚠️ **Important:** This is an academic/research prototype, **not a medical diagnostic tool**. It should not be used to diagnose skin cancer or replace a dermatologist or other qualified medical professional.

---

## ✨ Features

- 🖼️ Upload a skin-lesion image
- 🤖 Predict one of 7 lesion categories
- 📊 Display the top 3 predictions with probabilities
- ⚠️ Calculate a combined risk score for potentially concerning lesions
- 🔥 Generate a Grad-CAM visualization showing areas that influenced the prediction
- 🔍 Detect potentially blurry images
- 🌡️ Use temperature calibration to improve probability calibration
- 🎯 Use a calibrated threshold for risk flagging

---

## 🧠 Lesion Categories

The model was trained on the **HAM10000 dataset** and recognizes these 7 categories:

| Code | Category | Description |
|---|---|---|
| `akiec` | Actinic keratosis | A potentially pre-cancerous skin lesion |
| `bcc` | Basal cell carcinoma | A common type of skin cancer |
| `bkl` | Benign keratosis-like lesion | Generally benign |
| `df` | Dermatofibroma | Generally benign skin nodule |
| `mel` | Melanoma | A serious form of skin cancer |
| `nv` | Melanocytic nevus | Common mole |
| `vasc` | Vascular lesion | Lesions involving blood vessels |

### Risk grouping

The application treats these classes as potentially concerning:

```text
mel  → Melanoma
bcc  → Basal cell carcinoma
akiec → Actinic keratosis
```

The probabilities of these classes are combined into a single **risk score**.

---

# 🏗️ How It Works

The project has two main parts:

```text
              ┌────────────────────┐
              │   Input Image      │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Image Preprocessing│
              │ Resize + Normalize │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  EfficientNet-B0   │
              │   Image Model      │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ 7 Class Probabilities│
              └─────────┬──────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Top Predictions        Risk Score
                                   │
                                   ▼
                          Risk Flag / Result
                                   │
                                   ▼
                             Grad-CAM
```

---

# 📁 Project Structure

```text
skin-lesion/
│
├── app.py
├── model.pth
├── classes.json
├── config.json
├── requirements.txt
├── notebook23ba8f415c.ipynb
└── .gitignore
```

### `app.py`

The Streamlit application.

It handles:

- image uploading
- image preprocessing
- model loading
- prediction
- risk scoring
- result display
- Grad-CAM visualization

### `model.pth`

The trained **EfficientNet-B0** model weights.

### `classes.json`

Contains the 7 class labels used by the model.

### `config.json`

Contains the calibration settings:

- temperature
- risk flag threshold

### `requirements.txt`

Contains the Python packages required to run the application.

### `notebook23ba8f415c.ipynb`

The Kaggle notebook containing the training and evaluation pipeline.

---

# 🔬 Model

The project uses:

**EfficientNet-B0**

with **7 output classes**.

The model was trained using transfer learning on the **HAM10000 skin-lesion dataset**.

### Training pipeline

```text
HAM10000 Dataset
       │
       ▼
Patient/lesion-grouped split
       │
       ├── Training Set
       ├── Validation Set
       └── Test Set
       │
       ▼
Image Augmentation
       │
       ▼
EfficientNet-B0
       │
       ▼
Weighted Cross Entropy
       │
       ▼
Model Selection
       │
       ▼
Temperature Calibration
       │
       ▼
Risk Threshold Selection
       │
       ▼
model.pth
```

---

# 🧪 Dataset

This project uses the **HAM10000 ("Human Against Machine with 10000 training images") dataset**.

The dataset contains dermoscopic images of pigmented skin lesions across multiple diagnostic categories.

The training notebook uses lesion-level grouping when creating the train, validation, and test sets.

This is important because multiple images can belong to the same lesion. Grouping by `lesion_id` helps prevent images from the same lesion from appearing in different splits.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd skin-lesion
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

- PyTorch
- Torchvision
- timm
- Streamlit
- Pillow
- NumPy
- OpenCV
- Grad-CAM

---

# ▶️ Run the Application

From the project directory:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🖼️ Using the Application

### Step 1 — Upload an image

Upload a clear, well-lit image of a skin lesion.

Supported formats:

```text
.jpg
.jpeg
.png
```

### Step 2 — Image preprocessing

The application:

1. Converts the image to RGB
2. Resizes it to `256 × 256`
3. Center-crops it to `224 × 224`
4. Converts it into a tensor
5. Applies ImageNet normalization

### Step 3 — Model prediction

EfficientNet-B0 generates predictions for all 7 classes.

For example:

```text
Melanoma              42.1%
Melanocytic nevus     31.7%
Benign keratosis      12.8%
...
```

### Step 4 — Risk calculation

The application combines the probabilities of:

```text
Melanoma
+
Basal cell carcinoma
+
Actinic keratosis
```

to calculate the overall concerning-lesion score.

### Step 5 — Grad-CAM

The application generates a heatmap showing the areas that contributed to the model's prediction.

This is intended to make the prediction more interpretable.

---

# 📊 Model Calibration

The model uses **temperature scaling** after training.

The calibrated temperature is stored in:

```text
config.json
```

Example:

```json
{
  "temperature": 0.8041853904724121,
  "flag_threshold": 0.1
}
```

The temperature is applied to the model logits before calculating probabilities:

```python
probability = softmax(logits / temperature)
```

The risk threshold is then used to determine whether the combined concerning-class score should trigger a warning.

---

# 🔥 Grad-CAM

The application uses **Grad-CAM** to visualize where the model focused when making its prediction.

The output is shown as a heatmap over the input image.

This can help with model interpretability, but:

> A Grad-CAM heatmap does **not** prove that the highlighted region is medically meaningful or that the model's prediction is correct.

---

# ⚠️ Limitations

This project has several important limitations.

### 1. Not a medical diagnostic system

The model is an academic prototype and should not be used to make medical decisions.

A prediction from the application does not confirm or rule out cancer.

---

### 2. Training data vs real-world photographs

The model was trained using **dermoscopic images** from HAM10000.

A typical phone photograph is different from a dermoscopic image.

Therefore, performance on ordinary photographs may be significantly different from performance on the original dataset.

---

### 3. Limited number of conditions

The model only recognizes the following 7 categories:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

It is **not designed to recognize every type of skin condition**.

For example, it should not be assumed to detect:

- rashes
- fungal infections
- eczema
- psoriasis
- wounds
- other unrepresented skin diseases

---

### 4. Dataset diversity

The training dataset does not represent every skin tone, camera type, lighting condition, or clinical environment equally.

Performance can therefore vary across different populations and image conditions.

---

### 5. Biopsy confirmation

A model prediction cannot replace clinical examination or pathological confirmation.

If there is a concerning or changing lesion, a qualified healthcare professional should evaluate it.

---

# 🚀 Future Improvements

Possible future improvements include:

- [ ] Improve performance on real-world smartphone photographs
- [ ] Add image-quality assessment
- [ ] Improve class imbalance handling
- [ ] Compare EfficientNet with other architectures
- [ ] Add ROC and Precision-Recall curves
- [ ] Improve calibration evaluation
- [ ] Add more comprehensive model metrics
- [ ] Test performance across different skin tones
- [ ] Add model confidence estimation
- [ ] Improve Grad-CAM visualization
- [ ] Add automated experiment tracking
- [ ] Create a dedicated backend API
- [ ] Containerize the application
- [ ] Deploy the application

---

# 📚 References

### Dataset

**Tschandl, P., Rosendahl, C., & Kittler, H. (2018).**

The HAM10000 dataset:

> "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions."

Dataset commonly distributed through Kaggle as:

**Skin Cancer MNIST: HAM10000**

### Model

The project uses **EfficientNet-B0** through the `timm` library.

### Explainability

Grad-CAM is used to provide visual explanations of the model predictions.

---

# ⚖️ Disclaimer

This project is intended for **educational and research purposes only**.

It is not a medical device and has not been clinically validated.

The predictions produced by this application should not be considered a medical diagnosis.

If you are concerned about a skin lesion, especially one that is changing, bleeding, painful, or otherwise unusual, consult a qualified healthcare professional.

---

## 👨‍💻 Project Status

**Status:** Academic prototype

**Framework:** Streamlit

**Deep Learning:** PyTorch

**Model:** EfficientNet-B0

**Dataset:** HAM10000

**Explainability:** Grad-CAM

**Classes:** 7

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and sharing feedback or suggestions for improving the project.