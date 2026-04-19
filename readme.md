# Fake News Detection using Segmentation-Based BERT

## Setup Instructions
1. Clone the repo.
2. Place `Fake.csv` and `True.csv` inside a folder named `Data/`.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the script: `python your_script_name.py`




## Overview

This project implements a fake news detection system using a segmentation-based approach with BERT embeddings. The goal is to compare a standard whole-article BERT model with a more advanced method that analyzes internal structure within articles.

The key idea is that fake news articles may exhibit **semantic inconsistencies across different sections**, and modeling these differences can improve classification performance.

---

## Project Structure

Make sure your project is organized like this:
FINAL340WAPP/
│
├── main.py
├── README.md
└── Data/
├── Fake.csv
└── True.csv



The dataset files **must be inside the `Data/` folder**.

---

## Dataset

This project uses the **ISOT Fake News Dataset**.

Each CSV file contains:
- `title`
- `text`
- `subject`
- `date`

The code uses:
- `title`
- `text`

Labels are assigned as:
- Fake → `1`
- Real → `0`

---

## Methodology

### Baseline Model

- Combine `title + text`
- Generate a single embedding using DistilBERT
- Train Logistic Regression

### Segmentation-Based Model (Proposed)

1. Split each article into **3 segments**:
   - Beginning
   - Middle
   - End

2. Encode each segment using DistilBERT

3. Compute **cosine similarity** between segments:
   - Segment 1 ↔ Segment 2
   - Segment 2 ↔ Segment 3
   - Segment 1 ↔ Segment 3

4. Create final feature vector:
   - Segment embeddings
   - Similarity scores

5. Train Logistic Regression classifier

---

## Why This Approach

Traditional models treat an article as a single block of text. This project instead captures:

- Internal structure
- Narrative flow
- Semantic consistency

This allows the model to detect patterns that whole-text models may miss.

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers tqdm


Recommended Setup (Virtual Environment)
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn torch transformers tqdm
Windows
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn torch transformers tqdm
Running the Code

From the project root:

python main.py

or

python3 main.py


Hardware Support

The code automatically detects hardware:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Uses GPU if available
Falls back to CPU otherwise

No changes required.
