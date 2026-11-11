# Lung Cancer Risk Prediction
 
Train a Keras/TensorFlow neural network to predict lung cancer risk, using two [Kaggle](https://www.kaggle.com/) lung cancer datasets:
 
1. [Lung Cancer Prediction Dataset](https://www.kaggle.com/datasets/bharatxjain/lung-cance-prediction-dataset) → `lung_cancer.csv`
2. [Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) (survey data) → `survey_lung_cancer.csv`
## Project structure
 
```
ai/
├── LungCancerRegression.py   # Full pipeline: load → normalize → train → predict
├── lung_cancer.csv           # Dataset 1 (4,999 rows × 30 columns, no header row)
├── survey_lung_cancer.csv    # Dataset 2 (309 rows × 16 columns, with header row)
├── pyproject.toml            # Python dependencies
└── README.md
```
 
## The datasets
 
Both datasets contain health and lifestyle information used to predict whether an individual has lung cancer. Each row is one individual's record, and both are structured for binary classification, with the target being the presence or absence of lung cancer. They differ in size, features, and encoding, so they are **not interchangeable** without preprocessing.
 
| | `lung_cancer.csv` | `survey_lung_cancer.csv` |
|---|---|---|
| Rows | 4,999 | 309 |
| Columns | 30 (29 features + target) | 16 (15 features + target) |
| Header row | No | Yes |
| Target column | `lung_cancer_risk` (0/1) | `LUNG_CANCER` (`YES`/`NO`) |
| Class balance | 1,243 positive / 3,756 negative (~25/75) | 270 `YES` / 39 `NO` (~87/13) |
| Missing values | None | None |
| Duplicate rows | None | 33 |
| Source | Synthetic patient records | Survey responses |
 
### Dataset 1: `lung_cancer.csv`
 
4,999 synthetic patient records with 30 columns and **no header row**. Column names are assigned in the script via `pd.read_csv(..., names=[...])`.
 
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
 
### Dataset 2: `survey_lung_cancer.csv`
 
309 survey responses with 16 columns and a **header row**. Things to know before using it:
 
- `GENDER` is text (`M` / `F`) and `LUNG_CANCER` is text (`YES` / `NO`). Both must be encoded to numbers before training.
- Every column except `GENDER`, `AGE`, and `LUNG_CANCER` uses a **1 = No, 2 = Yes** encoding (not 0/1). Recode to 0/1 so it is consistent with Dataset 1.
- The headers `FATIGUE ` and `ALLERGY ` have a **trailing space**, and some names contain spaces (e.g. `CHRONIC DISEASE`). Strip and normalize the names after loading, e.g. `df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")`.
- It has **33 duplicate rows**; consider `df.drop_duplicates()`.
- It is **heavily imbalanced** (~87% `YES`), so accuracy alone is misleading. Check precision/recall too.
| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `GENDER` | M/F | Sex |
| 2 | `AGE` | int | Age (21–87) |
| 3 | `SMOKING` | 1/2 | Smoker |
| 4 | `YELLOW_FINGERS` | 1/2 | Yellow fingers |
| 5 | `ANXIETY` | 1/2 | Anxiety |
| 6 | `PEER_PRESSURE` | 1/2 | Peer pressure |
| 7 | `CHRONIC DISEASE` | 1/2 | Chronic disease |
| 8 | `FATIGUE ` | 1/2 | Symptom |
| 9 | `ALLERGY ` | 1/2 | Allergy |
| 10 | `WHEEZING` | 1/2 | Symptom |
| 11 | `ALCOHOL CONSUMING` | 1/2 | Alcohol consumption |
| 12 | `COUGHING` | 1/2 | Symptom |
| 13 | `SHORTNESS OF BREATH` | 1/2 | Symptom |
| 14 | `SWALLOWING DIFFICULTY` | 1/2 | Symptom |
| 15 | `CHEST PAIN` | 1/2 | Symptom |
| 16 | `LUNG_CANCER` | YES/NO | **Target** — 270 `YES` / 39 `NO` |

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

**NOTE:** If you already use an IDE with Python manager included (ie. PyCharm, Visual Studio, VS Code, ect) you can skip this step.

This project needs **Python 3.10 – 3.13** (TensorFlow publishes wheels for cp310–cp313; 3.14 is not yet supported).

**Windows**

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download a Python 3.12 or 3.13 installer (**NOT 3.14+** — no TensorFlow wheels yet).
2. Run the installer and **check "☑︎ Add python.exe to PATH"** on the first screen — if you miss this, `python` won't work in the terminal.
3. Verify: open Command Prompt (or PowerShell) and run:
   ```powershell
   py --list
   ```

**macOS**

Homebrew (recommended):
```bash
# Install Homebrew first if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.13
python3.13 --version
```

### Step 2 — Create a virtual environment (recommended)

Keeps this project's packages isolated from your system Python:

**Run in the folder containing LungCancerRegression.py:**

MacOS / Linux:
```shell
python3.13 -m venv .venv
source .venv/bin/activate
```

Windows:
```shell
py -3.13 -m venv .venv
.venv\Scripts\activate
```

Your prompt should now show `(venv)`.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install tensorflow tensorboard pandas numpy pydot graphviz
```

`plot_model` also needs the **Graphviz system binary** (the pip package alone is not enough):

**MacOS**
```bash
brew install graphviz
```

**Windows**, download the installer from [graphviz.org/download](https://graphviz.org/download/), run it, and add `C:\Program Files\Graphviz\bin` to your PATH (the installer offers an "Add to PATH" option).

> Note: the script contains a macOS-only PATH hack (`/opt/homebrew/bin`, line 10). On Windows/Linux you can delete that line once Graphviz is on your PATH.

### Step 4 — Activate the Virtual Environment in your IDE

If you are using Thonny:
1. Select `View > Files`
2. In the 'Files' sidebar open the project
3. There should be a folder called `.venv`
4. In the folder right click the file called `pyvenv.cfg`
5. Click 'Activate virtual environment'

### Step 5 — Run it

If you are using Thonny you can click the green play button

Or, run it from terminal 
```bash
python LungCancerRegression.py
```

First run downloads/loads TensorFlow and trains for 10 epochs — expect a minute or two on a laptop.

### Optional — watch training in TensorBoard

Run in terminal
```bash
tensorboard --logdir logs/fit
```
**then open [http://localhost:6006](http://localhost:6006)**

*Built as part of Lydian Academy STEM coursework.*
