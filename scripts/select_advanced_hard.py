
import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

ROOT = Path.home() / "reasoning-sft-gh200"
SRC = ROOT / "datasets/advanced_hard/data/acereason11_100k.json"
OUT = ROOT / "datasets/advanced_hard/advanced_hard15k.json"
STATS = ROOT / "datasets/advanced_hard/selection_stats.json"

N_SELECT = 15000
N_CLUSTERS = 100
MAX_TOKENS = 16384
DROP_SHORTEST = 0.50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("Loading dataset...", flush=True)
with open(SRC) as f:
    data = json.load(f)

print("Total examples:", len(data), flush=True)
stats = {"total": len(data)}

# 1. Measure full-example lengths using the Qwen tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    str(ROOT / "models/Qwen2.5-3B-Instruct")
)

def full_text(x):
    return "\n".join([
        x.get("system") or "",
        x["instruction"],
        x.get("input") or "",
        x["output"]
    ])

lengths = []

print("Calculating token lengths...", flush=True)
for start in range(0, len(data), 1000):
    batch = data[start:start + 1000]
    encoded = tokenizer(
        [full_text(x) for x in batch],
        add_special_tokens=False
    )["input_ids"]
    lengths.extend(len(ids) for ids in encoded)

    if start % 10000 == 0:
        print("Tokenized:", start, flush=True)

lengths = np.asarray(lengths)

# 2. Boxed-answer and length filters.
pool = [
    i for i, x in enumerate(data)
    if "\\boxed" in x["output"]
]
stats["after_boxed"] = len(pool)

pool = [
    i for i in pool
    if lengths[i] <= MAX_TOKENS - 64
]
stats["after_length"] = len(pool)

# 3. Remove shortest 50% from the eligible pool.
if not pool:
    raise RuntimeError("No eligible examples after filtering")

threshold = float(
    np.quantile(lengths[pool], DROP_SHORTEST)
)

pool = [
    i for i in pool
    if lengths[i] >= threshold
]

stats["length_threshold"] = threshold
stats["after_difficulty"] = len(pool)

# 4. Deduplicate problem statements.
random.shuffle(pool)

seen = set()
deduplicated = []

for i in pool:
    key = data[i]["instruction"].strip()

    if key not in seen:
        seen.add(key)
        deduplicated.append(i)

pool = deduplicated
stats["after_dedup"] = len(pool)

print("Filter statistics:", stats, flush=True)

if len(pool) < N_SELECT:
    raise RuntimeError(
        f"Only {len(pool)} eligible examples; "
        f"need {N_SELECT}. Do not silently relax filters."
    )

# 5. Embed problems and cluster into 100 groups.
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding device:", device, flush=True)

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

prompts = [
    data[i]["instruction"][:2000]
    for i in pool
]

embeddings = embedder.encode(
    prompts,
    batch_size=128,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Clustering...", flush=True)

cluster_model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    batch_size=4096,
    n_init=3
)

labels = cluster_model.fit_predict(embeddings)

clusters = defaultdict(list)

for i, label in zip(pool, labels):
    clusters[int(label)].append(i)

for cluster in clusters.values():
    random.shuffle(cluster)

# 6. Balanced cluster selection.
# Initial allocation: up to 150 from every cluster.
# Redistribute unused quota among clusters with capacity.
selected = []
remaining_clusters = {}

quota = N_SELECT // N_CLUSTERS

for label in sorted(clusters):
    members = clusters[label]
    take = min(quota, len(members))
    selected.extend(members[:take])
    remaining_clusters[label] = members[take:]

remaining = N_SELECT - len(selected)

while remaining > 0:
    available = [
        label for label in sorted(remaining_clusters)
        if remaining_clusters[label]
    ]

    if not available:
        raise RuntimeError("Insufficient samples for 15K")

    for label in available:
        if remaining == 0:
            break
        selected.append(remaining_clusters[label].pop())
        remaining -= 1

assert len(selected) == N_SELECT
assert len(set(selected)) == N_SELECT

random.shuffle(selected)

# 7. Save the selected dataset.
result = []

for i in selected:
    x = dict(data[i])
    x.pop("extra_info", None)
    result.append(x)

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w") as f:
    json.dump(result, f, ensure_ascii=False)

selected_lengths = lengths[selected]

stats.update({
    "selected": len(selected),
    "clusters": len(clusters),
    "selected_mean_tokens": float(selected_lengths.mean()),
    "selected_median_tokens": float(np.median(selected_lengths)),
    "selected_min_tokens": int(selected_lengths.min()),
    "selected_max_tokens": int(selected_lengths.max()),
    "all_mean_tokens": float(lengths.mean()),
    "all_median_tokens": float(np.median(lengths)),
    "seed": SEED
})

with open(STATS, "w") as f:
    json.dump(stats, f, indent=2)

print("\nSELECTION COMPLETED", flush=True)
print(json.dumps(stats, indent=2))
print("Dataset:", OUT)
print("Statistics:", STATS)
