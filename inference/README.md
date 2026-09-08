# Standalone Inference

Apply the original RDBPFN backbone to flat NumPy arrays or pandas DataFrames. This package is independent of the relational generation and training pipeline. It supports binary classification and one-vs-rest multiclass classification; it does not expose the fork's FK/entity attention biases.

## Install and Checkpoint

In an environment with a suitable PyTorch build:

```bash
python -m pip install numpy pandas scikit-learn torch
```

Run the examples below from `inference/`. `RDBPFNClassifier.from_pretrained("RDBPFN")` loads the local file `checkpoints/RDBPFN/model.pt`; it does not download a checkpoint. You can also pass a local checkpoint file or a directory containing `model.pt`. Checkpoints must match the fixed architecture in [src/checkpoint.py](src/checkpoint.py).

## Python Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.predictor import RDBPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)
clf = RDBPFNClassifier.from_pretrained("RDBPFN", device="cpu")
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)
print("ROC AUC", roc_auc_score(y_test, prob[:, 1]))
```

For DataFrames with a target column:

```python
import pandas as pd
from src.predictor import RDBPFNClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
clf = RDBPFNClassifier.from_pretrained("RDBPFN")
clf.fit(train_df, target="label")
features = test_df.drop(columns=["label"], errors="ignore")
prob = clf.predict_proba(features)
pred = clf.predict(features)
```

Prediction DataFrames must contain the same feature columns used for fitting. Numeric missing values are imputed from training data; categorical columns use training-derived ordinal encodings, with `-1` for missing or unseen categories. NumPy inputs must be numeric and have the same feature count. The default device is CUDA when available, otherwise CPU.

## CSV Command

```bash
python predict_csv.py \
  --train examples/demo_train.csv \
  --test examples/demo_test.csv \
  --target label \
  --checkpoint RDBPFN --device cpu \
  --output examples/demo_predictions.csv
```

Replace the example paths with your files. The target is required in the training table and optional in the test table. Output contains input features, `prediction`, and one probability column per class. Parquet inputs are also accepted when a pandas parquet engine such as `pyarrow` is installed.

The built-in demos are [demo.py](demo.py) and [demo_multiclass.py](demo_multiclass.py).

Prediction is automatically split into chunks of at most 2,000 test rows. This inference version fixes the chunk size at 2,000; passing a different `chunk_size` raises an error.

## Context Size

Each prediction context contains at most 1,024 training rows. The context builder uses:

| Available training rows | Contexts |
| --- | --- |
| At most 1,024 | One context using all rows. |
| 1,025–9,999 | `ceil(n / 1024)` sampled contexts; probabilities are averaged. |
| At least 10,000 | One sampled context of 1,024 rows. |

Multiclass prediction builds binary one-vs-rest contexts and normalizes the positive-class probabilities across classes. [src/predictor.py](src/predictor.py) defines the sampling and ensemble behavior.
