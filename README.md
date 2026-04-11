# Deep-Learning-2026-MDSAA

## Project Overview

This repository contains the implementation of a deep learning pipeline developed for the Deep Learning course (2025/2026). The primary objective is to address a predictive image classification problem using a modified subset of the **"wikiart"** dataset, which consists of scanned paintings from various authors. 

The project strictly utilises the **Keras** framework for model implementation and adheres to standard industry practices for developing generalisable predictive models. The core challenge involves capturing the underlying visual phenomena in the artwork data and selecting appropriate architectures to maximise classification performance.

---

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for Python environment management. The project is configured to automatically use Python 3.10 via the `.python-version` file.

### 1. Create the Environment

> **Note:** If you already see a `.venv` folder in your project directory, do not run `uv venv`. Skip directly to the activation step.

First, navigate to the project folder and create the virtual environment. `uv` will automatically detect the required Python version and download it if necessary:

```bash
uv venv
```

### 2. Activate the Environment

Next, activate the environment based on your operating system:

* **Windows:**
  ```bash
  .\.venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source .\.venv\Scripts\activate
  ```

### 3. Install Dependencies

Once the environment is activated, install the required deep learning packages (TensorFlow, Keras, etc.):

```bash
uv pip install -r requirements.txt
```

If your project is in an OneDrive folder, use:

```bash
uv pip install -r requirements.txt --link-mode=copy
```

---

## Methodology

The development lifecycle was structured into the following sequential phases:

### 1. Data Pipeline and Preparation

**Exploratory Data Analysis (EDA):** Analysis of class distributions, image dimensions, and dataset characteristics.

**Preprocessing & Splitting:** Image normalisation, resizing, and partitioning the dataset into training, validation, and testing subsets.

### 2. Custom Neural Network Architectures

**Custom Model:** Implementation of a custom Convolutional Neural Network (CNN) incorporating data augmentation techniques to improve generalisation and mitigate overfitting.

**Improved Model:** Design and training of the custom CNN without data augmentation as we realised there was no overfitting.

**Third Model:** Implementation of a third version of the CNN with some data augmentation.

### 3. Transfer Learning Approaches

**EfficientNetV2S:** Adaptation and fine-tuning of the EfficientNetV2S architecture.

**Xception:** Adaptation and fine-tuning of the Xception model for the specific artwork classification task.

### 4. State-of-the-Art Feature Extraction

**DinoV2 Integration:** Utilisation of the DinoV2 foundational model for advanced feature extraction, coupled with a custom Keras classifier for the final predictions.

### 5. Model Evaluation and Comparative Analysis

**Performance Metrics:** Systematic evaluation of all models.

**Trade-off Analysis:** Critical comparison of the computational cost, architectural complexity, and predictive accuracy among the custom networks, transfer learning models, and DinoV2.