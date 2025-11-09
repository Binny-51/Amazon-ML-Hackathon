# Amazon-ML-Hackathon
# 🧠 Memory-Efficient Multi-Modal Price Prediction Pipeline

### TF-IDF + SVD + EfficientNet-B0 + XGBoost

📦 **Challenge:** Amazon ML Hackathon 2025 – *Smart Product Pricing*
📊 **Goal:** Predict product prices from multimodal data (catalog text + product image)
⚡ **Local Metric (SMAPE):** ~36.8
🧮 **Frameworks:** PyTorch · timm · XGBoost · scikit-learn · Optuna

---

## 🚀 Overview

This repository implements a **memory-efficient, multimodal regression pipeline** to predict product prices using:

* **Text features** extracted via TF-IDF + TruncatedSVD
* **Image embeddings** from a pretrained EfficientNet-B0
* **XGBoost regression** optimized via Optuna

The focus is on:

* 🔹 Reducing RAM and GPU load
* 🔹 Maintaining strong model accuracy
* 🔹 Combining text + image modalities effectively

## 🧩 Problem Statement

### 🎯 **Business Objective**
E-commerce platforms must set **competitive and fair prices** for products listed on their marketplace.  
The objective is to **predict the price of a product** given its **structured catalog text** and an **associated image**.

Accurate price prediction enables:
- 💰 **Pricing recommendations** for sellers  
- 🤖 **Automated product listing and validation**  
- 🕵️ **Fraud detection** for outlier or misleading prices  
- 🔍 **Enhanced search ranking and relevance**

---

### 🧠 **Task Description (Machine Learning Perspective)**

#### **Input**
- **`catalog_content`** — A text blob containing structured fields such as:  
  - Item Name  
  - Unit  
  - Value  
  - Bullet Points  
  - Product Description  
- **`image_link`** — URL or local path to the corresponding product image  

#### **Output**
- A **predicted price** (positive floating-point number)

---

### 📊 **Evaluation Metric**

- **Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)  
- **Goal:** Lower SMAPE indicates better performance.  

\[
\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y_i}|}{(|y_i| + |\hat{y_i}|)/2}
\]

---

### ⚙️ **Constraints**

- 🚫 **No external price lookup** allowed (e.g., web scraping or third-party APIs)  
- 🧩 Only **open-source models/libraries** permitted (competition licensing constraints)  
- 🌐 The model should **generalize across**:  
  - Product categories  
  - Brands  
  - Image quality and style variations  

---

### 💡 **Why SMAPE?**

- SMAPE measures **relative error**, making it robust when product prices vary across **multiple orders of magnitude**.  
- It handles **imbalanced price distributions**, such as:  
  - Many low-cost items  
  - Few high-end or luxury items  


## 🧩 Architecture

```
Input: catalog_content, image_link, price
         │
         ▼
┌───────────────────────────────┐
│ Parse Catalog Content         │
│  • Item Name, Unit, Value     │
│  • Bullet Points & Description│
└───────────────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Clean & Normalize Text        │
│  • Remove stopwords, symbols  │
│  • Combine bullet pts + desc  │
└───────────────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ TF-IDF + TruncatedSVD         │
│  • Brand, Unit, Features      │
└───────────────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ EfficientNet-B0               │
│  • Extract 1280-D embeddings  │
│  • Batch processing w/ GPU    │
└───────────────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ XGBoost + Optuna              │
│  • SMAPE-based optimization   │
│  • GPU-accelerated training   │
└───────────────────────────────┘
         │
         ▼
 Output: Trained model + artifacts
```

---

## 🧱 Dataset

**Input Columns:**

| Column            | Description                                    |
| ----------------- | ---------------------------------------------- |
| `catalog_content` | Text block with structured product information |
| `image_link`      | URL or local path to product image             |
| `price`           | Target variable (float)                        |

**Example (simplified):**

```
Item Name: Amul Butter 500g
Unit: 500g
Value: 500
Bullet Point 1: Made from fresh cream
Product Description: Delicious, pure and healthy butter for everyday use.
Image Link: https://images.amazon.com/amul.jpg
Price: 250
```

---

## ⚙️ Configuration

| Parameter            | Value                           |
| -------------------- | ------------------------------- |
| Random Seed          | 42                              |
| Image Model          | EfficientNet-B0                 |
| Text Models          | TF-IDF (brand/features/unit)    |
| Dim Reduction        | TruncatedSVD (3000 + 5000 + 64) |
| ML Model             | XGBoost                         |
| Hyperparameter Tuner | Optuna (20 trials)              |
| Evaluation Metric    | SMAPE                           |
| Device               | GPU (cuda) if available         |

---

## 🧠 Key Components

### 1️⃣ **Text Parsing & Cleaning**

* Extracts fields like `Item Name`, `Unit`, `Value`, and `Product Description`
* Cleans and tokenizes text (lowercasing, regex removal, stopword filtering)
* Removes unit keywords (e.g., “kg”, “ml”) from item names

### 2️⃣ **TF-IDF + SVD Compression**

| Feature Type     | Max Features | SVD Components |
| ---------------- | ------------ | -------------- |
| Brand            | 15,000       | 3,000          |
| Product Features | 30,000       | 5,000          |
| Unit             | 128          | 64             |

Reduces sparse matrices into dense, low-memory representations.

### 3️⃣ **Image Embedding Extraction**

* Uses pretrained `EfficientNet-B0` (from `timm`)
* Generates **1280-D global average pooled embeddings**
* Batch processing (default: 200 images per batch)
* Handles missing or broken URLs gracefully (returns zero vector)

### 4️⃣ **Feature Fusion**

* Concatenates `[SVD-text vectors + scaled numeric value + image embeddings]`
* Results in a **dense multimodal feature matrix**

### 5️⃣ **Model Training (Optuna + XGBoost)**

Optuna optimizes:

```python
learning_rate ∈ [0.01, 0.04]
n_estimators ∈ [3000, 4000]
max_depth ∈ [6, 10]
```

Training uses GPU acceleration (`tree_method="gpu_hist"`).

**Early stopping:** 100 rounds
**Validation split:** 80/20
**Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)

---

## 📈 Results

| Model               | Data Used  | SMAPE (↓)  |
| :------------------ | :--------- | :--------- |
| TF-IDF + SVD        | Text only  | ~44.0      |
| EfficientNet        | Image only | ~41.5      |
| Text + Image Fusion | **Both**   | **36.8 ✅** |

---

## 💾 Saved Artifacts

| File                          | Description                     |
| ----------------------------- | ------------------------------- |
| `xgb_mem_efficient_model.pkl` | Final XGBoost model             |
| `vectorizers_svd.pkl`         | TF-IDF + SVD transformers       |
| `num_scaler.pkl`              | Scaler for numeric features     |
| `price_scaler.pkl`            | Scaler for target normalization |
| `image_features.npy`          | Cached EfficientNet embeddings  |

Saved automatically under:

```
models_mem_efficient_pipeline/
```

---

## 🔧 Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy torch timm xgboost optuna nltk scikit-learn pillow tqdm joblib requests
```

### 2️⃣ Prepare Dataset

Place your `train.csv` in the same directory:

```csv
catalog_content,image_link,price
"Item Name: Amul Butter 500g\nUnit: 500g\nValue: 500\nProduct Description: Fresh cream butter",https://...,250
```

### 3️⃣ Run the Pipeline

```bash
python train_mem_efficient_pipeline.py
```

### 4️⃣ Output

```
✅ Using device: cuda
🔹 Applying SVD (memory-efficient)...
Extracting Image Features in Batches: 100%
✅ Final combined feature shape: (N, ~9400)
✅ Best Params: {...}
✅ Best SMAPE: 36.8
✅ Model, SVD transformers, scalers & image features saved successfully!
```

---

## 🧩 Custom Functions

| Function                  | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| `parse_catalog_content()` | Parses structured text fields                   |
| `clean_text()`            | Token cleaning & normalization                  |
| `extract_img_features()`  | Extracts image embeddings (with error handling) |
| `smape()`                 | Custom evaluation metric                        |
| `objective()`             | Optuna trial function                           |

---

## 🧮 Performance Optimizations

* **SVD compression** drastically reduces TF-IDF memory usage
* **Batch image embedding extraction** prevents CUDA OOM
* **GPU-based XGBoost** for fast tree building
* **Optuna** automatically tunes hyperparameters efficiently
* **Log-transform + StandardScaler** improves regression stability

---

## 🔮 Future Scope

* Incorporate **CLIP** for unified image-text embedding
* Add **pseudo-labeling** for unlabeled products
* Explore **LightGBM + XGBoost stacking**
* Apply **quantization or pruning** for deployment efficiency

---

## 👨‍💻 Author

**Naman Agrawal**
IIT Bhubaneswar · Mechanical Engineering
📧 [[your.email@example.com](mailto:your.email@example.com)]
🔗 [github.com/yourusername]

---

Would you like me to make an additional **section with command examples for inference/prediction** (e.g., loading `xgb_mem_efficient_model.pkl` to predict on test.csv)?
That would complete this README for real submission use.
