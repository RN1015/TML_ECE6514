import json, os, random, collections
import numpy as np

HOME = os.path.expanduser("~")
SRC = f"{HOME}/data_raw/data/acereason11_100k.json"
OUT = f"{HOME}/LLaMA-Factory/data/advanced15k.json"
INFO = f"{HOME}/LLaMA-Factory/data/dataset_info.json"
STATS = f"{HOME}/advanced15k_stats.json"
N_SELECT, K, MAX_TOKENS, DROP_SHORTEST, SEED = 15000, 100, 16384, 0.25, 42

random.seed(SEED); np.random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
stats = {}

data = json.load(open(SRC))
n = len(data)
print("Loaded", n, "examples")
print("extra_info sample:", data[0].get("extra_info"))
stats["total"] = n

# 1) Token length of each full example (Qwen tokenizer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
def full_text(x):
    return "\n".join([x.get("system") or "", x["instruction"], x.get("input") or "", x["output"]])
lengths = []
for i in range(0, n, 1000):
    ids = tok([full_text(x) for x in data[i:i+1000]], add_special_tokens=False)["input_ids"]
    lengths.extend(len(t) for t in ids)
    if i % 10000 == 0:
        print("  tokenized", i)
lengths = np.array(lengths)

# 2) Filters: boxed answer, fits in cutoff_len
pool = [i for i in range(n) if "\\boxed" in data[i]["output"]]
stats["after_boxed_filter"] = len(pool)
pool = [i for i in pool if lengths[i] <= MAX_TOKENS - 64]
stats["after_length_filter"] = len(pool)

# 3) Drop shortest 25%
cut = float(np.quantile(lengths[pool], DROP_SHORTEST))
pool = [i for i in pool if lengths[i] >= cut]
stats["min_tokens_kept"] = cut
stats["after_drop_shortest"] = len(pool)

# Deduplicate identical prompts (keep one at random)
random.shuffle(pool)
seen, dedup = set(), []
for i in pool:
    key = data[i]["instruction"].strip()
    if key not in seen:
        seen.add(key); dedup.append(i)
pool = dedup
stats["after_dedup"] = len(pool)
print("Pool sizes:", {k: v for k, v in stats.items() if k != "total"})

# 4) Embed prompts and cluster
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Embedding on", dev)
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
emb = emb_model.encode([data[i]["instruction"][:2000] for i in pool], batch_size=256,
                       show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
labels = MiniBatchKMeans(n_clusters=K, random_state=SEED, batch_size=4096, n_init=3).fit_predict(emb)

clusters = collections.defaultdict(list)
for i, c in zip(pool, labels):
    clusters[int(c)].append(i)
for c in clusters.values():
    random.shuffle(c)

# Balanced sampling: smallest clusters first, spread remaining quota evenly
selected, remaining = [], N_SELECT
order = sorted(clusters, key=lambda c: len(clusters[c]))
for j, c in enumerate(order):
    quota = remaining // (len(order) - j)
    take = clusters[c][:quota]
    selected.extend(take); remaining -= len(take)
print("Selected", len(selected))

# Save (without extra_info) and register
out = []
for i in selected:
    x = dict(data[i]); x.pop("extra_info", None); out.append(x)
json.dump(out, open(OUT, "w"), ensure_ascii=False)
info = json.load(open(INFO))
info["advanced15k"] = {"file_name": "advanced15k.json"}
json.dump(info, open(INFO, "w"), indent=2, ensure_ascii=False)

sel_len = lengths[selected]
stats.update({
    "selected": len(selected),
    "cluster_sizes_in_pool": sorted(len(v) for v in clusters.values()),
    "selected_tokens_mean": float(sel_len.mean()),
    "selected_tokens_median": float(np.median(sel_len)),
    "all_tokens_mean": float(lengths.mean()),
    "all_tokens_median": float(np.median(lengths)),
})
json.dump(stats, open(STATS, "w"), indent=2)
print("Saved", OUT, "and registered 'advanced15k'. Stats in", STATS)
