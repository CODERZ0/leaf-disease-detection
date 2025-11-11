# compute_class_weights.py
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json

train_dir = "data/train"
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
counts = []
for c in classes:
    n = len([f for f in os.listdir(os.path.join(train_dir, c)) if os.path.isfile(os.path.join(train_dir, c, f))])
    counts.append(n)
    print(f"{c} : {n}")

# prepare labels vector for compute_class_weight
y = []
for i, n in enumerate(counts):
    y += [i] * n
y = np.array(y)

cw = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {int(i): float(w) for i, w in enumerate(cw)}
print("\nclass_weight computed (first 10 shown):")
print(dict(list(class_weight.items())[:10]))
with open("class_weight.json", "w") as f:
    json.dump(class_weight, f)
print("\nSaved class_weight.json")
