# Spam Detection Using Machine Learning Classifiers

A comparative machine-learning study of spam and non-spam email classification using the UCI Spambase dataset. The assignment evaluates Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) using stratified 10-fold cross-validation.

## Assignment overview

The objective is to compare the classifiers from two perspectives:

- **Predictive performance:** accuracy and F1-score.
- **Computational performance:** model training time.

The assignment also applies the Friedman and Nemenyi procedures to compare classifier rankings across the cross-validation folds.

## Algorithms evaluated

| Classifier | Description in this project |
|---|---|
| Logistic Regression (LR) | Linear classification baseline. |
| Support Vector Machine (SVM) | Support-vector classifier using scikit-learn's `SVC`. |
| K-Nearest Neighbors (KNN) | Neighbour-based classifier. |

## Dataset

The project uses the **UCI Spambase** dataset. It contains 57 numerical attributes extracted from email messages and a binary class label identifying an email as spam or non-spam.

Dataset source: [UCI Machine Learning Repository — Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

The dataset files are not included in this repository. Download `spambase.data` and `spambase.names` from the UCI dataset page and place both files in the project root, alongside the notebook.

## Methodology

1. Read the feature names from `spambase.names` and load the observations from `spambase.data`.
2. Separate the input attributes from the target label.
3. Standardize the input features with scikit-learn's `StandardScaler`.
4. Evaluate LR, SVM, and KNN using `StratifiedKFold` with 10 folds, shuffling enabled, and `random_state=42`.
5. Record training time, accuracy, and F1-score for each fold.
6. Rank the classifiers for each metric and perform the Friedman and Nemenyi comparisons implemented in the notebook.

## Results reported in the assignment

The following values are the average results reproduced from the submitted report. They are reported here as assignment results, not as a separate re-run of the experiments.

| Classifier | Mean training time (s) | Mean accuracy | Mean F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.0640 | 0.9255 | 0.9036 |
| Support Vector Machine | 1.1701 | 0.9339 | 0.9145 |
| K-Nearest Neighbors | 0.0278 | 0.9092 | 0.8834 |

In the reported averages, SVM has the highest accuracy and F1-score, while KNN has the shortest training time. The complete fold-level tables and the statistical discussion are available in `ML_Assign-2_Report.pdf`.

**Statistical-results note:** The submitted report contains inconsistent statements about the interpretation of the Friedman-test p-values and the Nemenyi comparisons. The README therefore presents the numerical averages above without making a new claim about statistical significance; consult the report and notebook for the original analysis.

## Repository structure

```text
Machine-Learning-Spam-Detection/
├── README.md
├── requirements.txt
├── spam_detection.ipynb
└── ML_Assign-2_Report.pdf
```

## Requirements

- Python 3.9 or later
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- scikit-learn
- SciPy

Install the Python dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Run the notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/santhosh-1414/Machine-Learning-Spam-Detection.git
   cd Machine-Learning-Spam-Detection
   ```

2. Download `spambase.data` and `spambase.names` from the [UCI Spambase dataset page](https://archive.ics.uci.edu/dataset/94/spambase) and place them in the project root.
3. Install the requirements:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `spam_detection.ipynb` and run the cells from top to bottom.

The notebook uses relative dataset filenames (`spambase.data` and `spambase.names`), so it should be run with the project root as the working directory.

## Files

- `spam_detection.ipynb` — implementation of the classifiers, cross-validation evaluation, metric ranking, and statistical tests.
- `ML_Assign-2_Report.pdf` — submitted assignment report, including the methodology, results tables, statistical discussion, and references.
- `requirements.txt` — Python package dependencies.

## Authors

**Santhosh Birada** and **Ayyappa Mitta**  
Computer Science  
Blekinge Institute of Technology (BTH)  
Karlskrona, Sweden
