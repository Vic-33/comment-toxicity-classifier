# Comment Toxicity Classifier 

A multiclass text classification project that predicts the **toxicity category of user
comments** (4 classes) from comment text plus structured metadata, built entirely with
**classical machine learning models** as required by the competition rules — no neural
networks / deep learning.

**Author:** K Balvivek Reddy · balvivekreddy33@gmail.com

**Competition:** `comment-category-prediction-challenge` (Private Kaggle Challenge)

## Problem Statement

Given a comment's text and accompanying metadata (voting stats, poster demographics,
timestamp, interaction flags), predict which of 4 categories it belongs to:

| Label | Interpretation |
|---|---|
| 0 | Neutral |
| 1 | Identity-based toxicity |
| 2 | General toxicity |
| 3 | Severe toxicity |

**Constraint:** Only classical/basic ML models are allowed (Logistic Regression, Random
Forest, gradient-boosted trees, etc.) — neural network-based approaches are explicitly
out of scope for this competition.

## Repository Contents

| File | Description |
|---|---|
| `notebook.ipynb` | End-to-end analysis: EDA → feature engineering → preprocessing → model training/comparison → submission. |
| `train.csv` / `test.csv` / `Sample.csv` | Kaggle-provided data (not included here — see [Data](#data)). |
| `submission.csv` | Generated predictions in the competition's required format (produced by running the notebook). |

## Data

The training set includes:
- `comment` — free-text comment (primary signal)
- `created_date` — UTC timestamp of posting
- `race`, `religion`, `gender`, `disability` — sparsely populated demographic/context columns
- `upvote`, `downvote`, `if_1`, `if_2`, `emoticon_1/2/3` — engagement and interaction counters
- `post_id` — row identifier
- `label` — target (train only; predicted for test)

The test set mirrors this schema minus `label`. `Sample.csv` defines the required
submission format (one `label` per row).

> Data files are provided by the Kaggle competition and are not redistributed in this
> repo — place `train.csv`, `test.csv`, and `Sample.csv` under
> `/kaggle/input/competitions/comment-category-prediction-challenge/` (or update the
> paths in the loading cell) before running.

## Notebook Walkthrough

1. **Data Loading** — reads train/test/sample files, initial shape and structure checks.
2. **Exploratory Data Analysis**
   - Missing value audit (train vs. test) — `race`, `religion`, `gender`, `disability` carry the most nulls, consistently across both splits.
   - Cardinality check — flags the near-unique `comment` column and any zero-variance columns.
   - Target distribution — confirms class imbalance (label 3 is rare, label 1 is the hardest to separate from neutral).
   - Outlier detection (IQR method) on numeric columns — documented but not removed, since the tree-based models used downstream are outlier-robust.
   - Numerical feature analysis — correlation with target, univariate distributions, bivariate plots.
   - Categorical feature analysis — univariate counts and per-category average label value.
   - Skewness analysis — identifies right-skewed count features for log transformation.
3. **Feature Engineering**
   - Log1p transform on skewed positive count features (`upvote`, `downvote`, `if_1`, `if_2`).
   - Temporal features from `created_date` (hour, day-of-week, month, year, is_weekend).
   - Text-derived structural features from `comment` (length, word count, avg word length, exclamation/question counts, caps ratio, URL presence).
   - Interaction/aggregate features (`vote_ratio`, `total_votes`, `if_sum`, `emoticon_sum`).
   - Justified column drops (raw `created_date`, any zero-variance columns) — `comment` is dropped only at the preprocessing stage once TF-IDF has consumed it.
4. **Preprocessing Pipeline**
   - `ColumnTransformer` routing: numeric columns → median imputation + `StandardScaler`; categorical columns → constant imputation + `OneHotEncoder(handle_unknown='ignore')`.
   - Hybrid **TF-IDF** text vectorization: word-level (unigrams+bigrams, 40k features) and character-level (`char_wb`, 3–5-grams, 50k features) to capture both vocabulary and misspelling/obfuscation patterns.
   - Word TF-IDF + char TF-IDF + structured features stacked into one sparse matrix (`scipy.sparse.hstack`).
   - Stratified 80/20 train/validation split to preserve class balance.
5. **Model Building** — three classical models trained and compared:
   - **Logistic Regression** (multinomial) — fast, interpretable linear baseline.
   - **Random Forest** — captures non-linear interactions, gives Gini-based feature importance.
   - **LightGBM** — gradient-boosted trees; native sparse-matrix support made it the fastest and best-performing of the three.
   - For each model: baseline fit → feature-importance-based feature selection experiment → hyperparameter tuning via `RandomizedSearchCV`.
6. **Model Comparison** — baseline vs. tuned validation accuracy compared side by side; LightGBM selected as the final model.
7. **Best Model — LightGBM** — retrained on the full training set (train + validation combined) and evaluated with a confusion matrix and classification report.
8. **Submission** — predictions generated on the test set and written to `submission.csv` in the competition's required format.

## Results

Final tuned **LightGBM** classifier, validation set:

| Metric | Value |
|---|---|
| Overall accuracy | **0.916** |
| Label 0 (Neutral) — precision / recall / F1 | 0.98 / 0.95 / 0.96 |
| Label 1 (Identity toxicity) — precision / recall / F1 | 0.78 / 0.78 / 0.78 |
| Label 2 (General toxicity) — precision / recall / F1 | 0.86 / 0.92 / 0.89 |
| Label 3 (Severe toxicity) — precision / recall / F1 | 0.76 / 0.53 / 0.63 |

LightGBM outperformed both Logistic Regression and Random Forest at every stage of
comparison, largely due to its native handling of the large sparse TF-IDF feature space
and leaf-wise tree growth. The model performs strongly on the majority classes (0, 2)
and is comparatively weaker on the minority classes (1, 3) — a direct consequence of
class imbalance, with most confusion occurring between semantically adjacent classes
(2↔0, 2↔1).

## A Note on Commented-Out Cells

Several cells in the notebook are intentionally commented out rather than deleted, to
keep the full development process visible while avoiding a slow end-to-end re-run:

- **`RandomizedSearchCV` hyperparameter searches** (Logistic Regression, Random Forest,
  LightGBM) — these were run once during development; their best-found parameters are
  written as comments directly above the commented search code, and the model actually
  used further down is refit with those parameters. Re-enabling the search cells
  reproduces the same tuning process from scratch.
- **The entire LightGBM baseline/feature-importance section** — commented out purely to
  save computation time on repeated notebook runs, since the final tuned LightGBM model
  (used for the submission) is trained independently later in the notebook.
- **The final validation confusion matrix / classification report** — the confusion
  matrix values and classification report for the tuned LightGBM model are hardcoded
  from a completed validation run rather than recomputed live, again to avoid re-running
  the full training pipeline just to display results already obtained.
- **Cross-validation cells** — commented out for the same reason: cross-validation
  scoring is useful during model development but adds significant runtime without
  changing the final model choice, so it's preserved as reference code rather than run
  on every execution.

To reproduce results from scratch (rather than trusting the recorded/hardcoded values),
uncomment the relevant cells and re-run — the notebook is fully functional either way.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
   ```
2. Place the competition's `train.csv`, `test.csv`, and `Sample.csv` at the path
   referenced in the data-loading cell (or edit the paths to match your environment).
3. Run the notebook top to bottom. By default it uses the pre-tuned hyperparameters and
   hardcoded validation metrics to keep runtime short; uncomment the tuning/CV cells if
   you want to reproduce the search process itself.
4. The final cell writes `submission.csv` in the format expected by the competition.

## Key Design Decisions

- **No neural networks** — per competition rules, the model search space was restricted
  to classical ML (linear, bagging, boosting), which shaped both the feature engineering
  (hand-crafted text/interaction features matter more without a network to learn them
  implicitly) and the modeling approach (TF-IDF instead of learned embeddings).
- **Hybrid word + character TF-IDF** — chosen specifically to catch obfuscated or
  misspelled toxic language that word-level features alone would miss.
- **Median imputation + robust scaling choices** — driven directly by the outlier
  analysis in EDA, since several numeric columns (votes, interaction flags) are heavily
  right-skewed.
- **Stratified splitting throughout** — necessary given the confirmed class imbalance,
  to avoid a validation set that misrepresents minority-class performance.
