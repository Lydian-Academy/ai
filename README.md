# Lung Cancer Risk Prediction
[New dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)

A beginner-friendly deep learning project that trains a Keras/TensorFlow neural network to predict lung cancer risk from 29 patient features. Using [kaggle Lung Cance Prediction Dataset](https://www.kaggle.com/datasets/bharatxjain/lung-cance-prediction-dataset)

## Project structure

```
ai/
├── LungCancerRegression.py   # Full pipeline: load → normalize → train → predict
├── lung_cancer.csv           # Dataset (4,999 rows × 30 columns, no header row)
├── requirements.txt          # Python dependencies
└── README.md
```

## The dataset

The Lung Cancer dataset contains health and lifestyle information used to predict whether an individual has lung cancer. It includes both demographic and clinical features related to smoking habits, respiratory symptoms, and general health conditions.

Each row represents a unique individual's record, and the dataset is structured for binary classification with the target variable being the presence or absence of lung cancer.
`lung_cancer.csv` contains 4,999 synthetic patient records with 30 columns and **no header row** — column names are assigned in the script via `pd.read_csv(..., names=[...])`.

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `age` | int | Patient age (18–90) |
| 2 | `gender` | 0/1 | Sex indicator |
| 3 | `education_years` | int | Years of education |
| 4 | `income_level` | int | Income bracket |
| 5 | `smoker` | 0/1 | Current/former smoker |
| 6 | `smoking_years` | int | Years of smoking |
| 7 | `cigarettes_per_day` | int | Daily cigarette count |
| 8 | `pack_years` | int | Pack-year smoking history |
| 9 | `passive_smoking` | 0/1 | Second-hand smoke exposure |
| 10 | `air_pollution_index` | int | Ambient air pollution level |
| 11 | `occupational_exposure` | 0/1 | Workplace carcinogen exposure |
| 12 | `radon_exposure` | 0/1 | Radon exposure |
| 13 | `family_history_cancer` | 0/1 | Family cancer history |
| 14 | `copd` | 0/1 | COPD diagnosis |
| 15 | `asthma` | 0/1 | Asthma diagnosis |
| 16 | `previous_tb` | 0/1 | Previous tuberculosis |
| 17 | `chronic_cough` | 0/1 | Symptom |
| 18 | `chest_pain` | 0/1 | Symptom |
| 19 | `shortness_of_breath` | 0/1 | Symptom |
| 20 | `fatigue` | 0/1 | Symptom |
| 21 | `bmi` | int | Body mass index |
| 22 | `oxygen_saturation` | int | SpO₂ (%) |
| 23 | `fev1_x10` | int | Lung function (FEV₁ × 10) |
| 24 | `crp_level` | int | Inflammation marker |
| 25 | `xray_abnormal` | 0/1 | Abnormal chest X-ray |
| 26 | `exercise_hours_per_week` | int | Weekly exercise |
| 27 | `diet_quality` | int | Diet quality score |
| 28 | `alcohol_units_per_week` | int | Weekly alcohol units |
| 29 | `healthcare_access` | int | Access-to-care score |
| 30 | `lung_cancer_risk` | 0/1 | **Target** — 1,243 positive / 3,756 negative (~25/75 split) |

## The model

A feedforward binary classifier:

```
Input (29 features)
 → Dense(128, relu)
 → Dense(64, relu)
 → Dense(32, relu)
 → Dense(1, sigmoid)   # probability of lung cancer risk
```

- **Loss:** `MeanSquaredError`
- **Optimizer:** `Adam`
- **Training:** 10 epochs, batch size 64, with a TensorBoard callback (`logs/fit/<timestamp>/`)
- **Outputs:** model summary, training history, a `lung_cancer_model.png` architecture diagram, and three example predictions

## Getting started

### Step 1 — Install Python

If you already use an IDE with Python included (ie. PyCharm, Thonny, Visual Studio, VS Code, ect) you can skip this step. If you aren't using an IDE **install** [Thonny](https://thonny.org/).

This project needs **Python 3.10 – 3.13** (TensorFlow publishes wheels for cp310–cp313; 3.14 is not yet supported).

<details>

<summary>Manually install Python interpreter (optional)</summary>

**Windows**

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download a Python 3.12 or 3.13 installer (avoid 3.14 — no TensorFlow wheels yet).
2. Run the installer and **check "☑︎ Add python.exe to PATH"** on the first screen — if you miss this, `python` won't work in the terminal.
3. Verify: open Command Prompt (or PowerShell) and run:
   ```powershell
   python --version
   ```

**macOS**

Option A — Homebrew (recommended):
```bash
# Install Homebrew first if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.13
python3 --version
```

Option B — official installer: download the macOS package from [python.org/downloads](https://www.python.org/downloads/) and double-click it. macOS also ships a system Python, but don't rely on it — use your own install/Thonny's install.

</details>

### Step 2 — Create a virtual environment (recommended)

Keeps this project's packages isolated from your system Python:

**Run in the folder containing LungCancerRegression.py:**
`python -m venv .venv`            # Windows
`python3 -m venv .venv`           # macOS / Linux

.venv\Scripts\activate          # activate on Windows
source .venv/bin/activate       # activate on macOS / Linux
```

Your prompt should now show `(venv)`.

### Step 3 — Install dependencies

All Python packages are pinned in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verified working sets (Linux x86_64, full pipeline run end-to-end): **Python 3.12 / TensorFlow 2.20.0 / numpy 2.2.6** and **Python 3.13 / TensorFlow 2.21.0 / numpy 2.5.4** (with pandas 3.1.0, pydot, graphviz, tensorboard).

`plot_model` also needs the **Graphviz system binary** (the pip package alone is not enough):

**macOS**
```bash
brew install graphviz
```

**Windows**, download the installer from [graphviz.org/download](https://graphviz.org/download/), run it, and add `C:\Program Files\Graphviz\bin` to your PATH (the installer offers an "Add to PATH" option).

> Note: the script contains a macOS-only PATH hack (`/opt/homebrew/bin`, line 10). On Windows/Linux you can delete that line once Graphviz is on your PATH.

### Step 4 — Point the script at your data

Change it to wherever your copy lives — or, if `lung_cancer.csv` sits next to the script, simply use:

```python
"lung_cancer.csv"
```

### Step 5 — Run it

```bash
python LungCancerRegression.py
```

First run downloads/loads TensorFlow and trains for 10 epochs — expect a minute or two on a laptop.

### Optional — watch training in TensorBoard

```bash
tensorboard --logdir logs/fit
```
**then open [http://localhost:6006](http://localhost:6006)**

*Built as part of Lydian Academy STEM coursework.*
