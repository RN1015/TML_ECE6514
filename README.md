# Reasoning SFT of Qwen2.5-3B-Instruct: Does Smarter Data Selection Beat Random?

**ECE 6514 Trustworthy Machine Learning, Group 3**
Mahesha Viduranga Malmalabaduge · Veera Manikantha Rayudu Tummala · Ishtiaque Ahmed Toke

We fully fine-tune **Qwen2.5-3B-Instruct** on 15K examples drawn from the course's 100K subset of **AceReason-1.1-SFT**, and compare a **random** 15K subset against two **curated** subsets (filtering + topic-balanced clustering). All models are evaluated on 8 math and science reasoning benchmarks with the course evaluation suite.

**Main result:** all three fine-tuned models improve over the base model on pass@8. **The random subset gave the best model** (8-benchmark average Acc 0.568 vs 0.540 for the base model). Our curated subsets did not beat random. We attribute this mainly to loss of domain diversity (see [Findings](#findings)).

**Leaderboard submission:** [`mahesha-vt/qwen25-3b-instruct-random15k`](https://huggingface.co/mahesha-vt/qwen25-3b-instruct-random15k)

---

## Models and datasets on Hugging Face

| Run | Model | Training data |
|---|---|---|
| Random (baseline) | [mahesha-vt/qwen25-3b-instruct-random15k](https://huggingface.co/mahesha-vt/qwen25-3b-instruct-random15k) | [mahesha-vt/acereason-random15k](https://huggingface.co/datasets/mahesha-vt/acereason-random15k) |
| Advanced (drop shortest 25% + clusters) | [IshtiaqueAhmedToke/qwen25-3b-advanced15k](https://huggingface.co/IshtiaqueAhmedToke/qwen25-3b-advanced15k) | [IshtiaqueAhmedToke/acereason-advanced15k](https://huggingface.co/datasets/IshtiaqueAhmedToke/acereason-advanced15k) |
| Advanced-hard (drop shortest 50% + clusters) | [rn1015/qwen25-3b-advanced-hard15k-bf16](https://huggingface.co/rn1015/qwen25-3b-advanced-hard15k-bf16) | rebuild with `scripts/select_advanced_hard.py` (seed 42) |

---

## Repository layout

```
configs/                      LLaMA-Factory training configs (one per model)
  random15k_1gpu.yaml
  advanced15k_1gpu.yaml
  advanced_hard15k_1gpu.yaml
scripts/                      Data selection
  make_random.py              random 15K subset (seed 42)
  select_advanced.py          advanced: filters + 100-cluster balanced sampling
  select_advanced_hard.py     advanced-hard: same, but drops the shortest 50%
datasets/<run>/selection_stats.json   counts after each selection step
results/<run>/                training summaries, loss curves, trainer state
results/<run>/eval/log_*.txt  evaluation logs (Acc and Pass@1 at the end of each file)
Project-Reasoning-SFT-LLM/    course repo; eval/eval_local.sh and eval/eval_single_local.sh
                              are our non-SLURM versions of the course eval scripts
```

Model weights, checkpoints and dataset JSON files are **not** stored in this repo (see `.gitignore`). They are on Hugging Face (links above).

---

## Data selection

All subsets are drawn from [`redsgnaoh/acereason11_100k`](https://huggingface.co/datasets/redsgnaoh/acereason11_100k) (100,000 examples in LLaMA-Factory format). Every subset has exactly **15,000** examples, and the `extra_info` field is removed.

### Random (baseline): `scripts/make_random.py`
`random.seed(42)`, then `random.sample(data, 15000)`. No filtering, so the subset keeps the pool's natural mix of math, code and science, including duplicates and over-long examples.

### Advanced: `scripts/select_advanced.py`
1. **Answer-format filter:** keep only solutions containing `\boxed{}`, since all benchmarks are graded on the boxed answer. This removes code-only solutions.
2. **Length filter:** keep examples of at most 16,320 tokens (Qwen tokenizer), so nothing is truncated at `cutoff_len = 16384` and loses its final answer.
3. **Difficulty proxy:** drop the shortest **25%** of the remaining solutions (very short solutions are usually easy problems).
4. **Deduplicate** identical problem statements.
5. **Topic diversity:** embed each problem with `sentence-transformers/all-MiniLM-L6-v2`, cluster into **100 groups** with MiniBatch K-Means, and sample **evenly across clusters** (150 per cluster).

### Advanced-hard: `scripts/select_advanced_hard.py`
Identical to Advanced, except step 3 drops the shortest **50%**.

### Selection funnel

| Step | Advanced | Advanced-hard |
|---|---|---|
| Start | 100,000 | 100,000 |
| Has `\boxed{}` answer | 67,440 | 67,440 |
| Fits in 16,384 tokens | 64,842 | 64,842 |
| Drop shortest 25% / 50% | 48,631 (≥ 3,710 tokens) | 32,424 (≥ 6,486 tokens) |
| Deduplicate prompts | 43,563 | 29,437 |
| Cluster-balanced sample | **15,000** | **15,000** |

| Example length (tokens) | Mean | Median |
|---|---|---|
| Full 100K pool | 7,508 | 6,650 |
| Advanced 15K | 8,699 | 8,017 |
| Advanced-hard 15K | 10,503 | 10,083 |

Observations: about **1/3 of the pool has no boxed answer** (mostly code), and about **10% of problem statements are duplicates**. For Advanced, every one of the 100 clusters had at least 221 candidates, so each contributed exactly 150 examples. A random sample would follow the natural cluster sizes (221–922).

---

## Training

Full-parameter SFT with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (v0.9.6.dev0). **Hyperparameters are identical for all runs**, so the only intended difference between models is the training data.

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Fine-tuning | full parameters, `template: qwen`, bf16 |
| Max sequence length | 16,384 |
| Learning rate | 5e-6, cosine schedule, no warmup |
| Batch size | 1 per device × 8 gradient accumulation = **8** |
| Epochs / steps | 2 / 3,750 |
| Attention | `flash_attn: auto` (falls back to PyTorch SDPA) |

| Run | Hardware | Memory strategy | Final train loss | Time |
|---|---|---|---|---|
| Random | 1× A100 80GB | DeepSpeed ZeRO-3 | 0.605 | 22.5 h |
| Advanced | 1× A100 80GB | DeepSpeed ZeRO-3 | 0.558 | 25.6 h |
| Advanced-hard | 1× GH200 (96 GB) | gradient checkpointing, no DeepSpeed | 0.569 | 10.1 h |

Training losses are not directly comparable across runs because each run trains on different data. Loss curves are in `results/<run>/training_loss.png`.

---

## Evaluation

Course evaluation suite (`Project-Reasoning-SFT-LLM/eval`, vLLM generation + rule-based grading of `\boxed{}` answers) with the default leaderboard settings: temperature 0.6, top-p 0.95, **8 samples per problem**, max 32,768 tokens, system prompt *"Please reason step by step, and put your final answer within \boxed{}."*

**All four models were evaluated on the same GH200 with the same settings.** Only vLLM's `gpu_memory_utilization` differed between runs, which affects speed, not outputs.

- **Acc** = pass@8: a problem counts as solved if any of the 8 samples is correct.
- **Pass@1** = average accuracy over the 8 samples.

### Results (Acc / Pass@1)

| Benchmark | # | Base | Random | Advanced | Advanced-hard |
|---|---|---|---|---|---|
| AIME 2024 | 30 | **0.233** / **0.075** | **0.233** / 0.050 | 0.133 / 0.033 | 0.200 / 0.063 |
| AMC 2023 | 40 | 0.700 / 0.394 | **0.725** / 0.388 | 0.700 / 0.381 | **0.725** / **0.406** |
| CN Math 2024 | 30 | 0.300 / 0.104 | **0.333** / 0.100 | **0.333** / **0.146** | **0.333** / 0.104 |
| GPQA-Diamond | 198 | 0.737 / **0.299** | **0.768** / 0.248 | 0.758 / 0.248 | 0.727 / 0.252 |
| Kaoyan | 199 | 0.528 / 0.221 | **0.648** / **0.249** | 0.593 / 0.239 | 0.608 / 0.226 |
| MATH-500 | 500 | 0.858 / 0.652 | **0.870** / 0.651 | 0.866 / **0.660** | 0.866 / 0.654 |
| Minerva Math | 272 | 0.456 / **0.309** | **0.467** / 0.276 | 0.460 / 0.279 | 0.460 / 0.279 |
| OlympiadBench | 675 | 0.508 / 0.284 | 0.499 / 0.281 | 0.513 / 0.283 | **0.523** / **0.290** |
| **Average (8)** | | 0.540 / **0.292** | **0.568** / 0.280 | 0.544 / 0.284 | 0.555 / 0.284 |
| **Average (7, excl. MATH-500)** | | 0.495 / **0.241** | **0.525** / 0.227 | 0.499 / 0.230 | 0.511 / 0.231 |

The 7-benchmark average leaves out MATH-500, following the averaging used by the course leaderboard. All comparisons are against our own base model, evaluated with the same protocol.

---

## Findings

1. **Fine-tuning helps pass@8.** All three fine-tuned models beat the base model on the 8-benchmark Acc average (+0.004 to +0.028). The largest gain is on Kaoyan (+0.07 to +0.12).
2. **Random was the best subset.** It has the highest 8- and 7-benchmark Acc averages (0.568 / 0.525), ahead of Advanced-hard (0.555 / 0.511) and Advanced (0.544 / 0.499).
3. **Likely cause: loss of domain diversity.** Both curated subsets are math-only (the `\boxed{}` filter removes code and most non-math content). Random keeps the full mix and has the best GPQA-Diamond (science) Acc, 0.768. Balancing topics *within* math did not compensate for losing variety *across* domains.
4. **Harder data helps on hard problems.** Advanced-hard (drop shortest 50%) beats Advanced (drop 25%) on average, and is the best model on OlympiadBench (0.523 / 0.290), the hardest benchmark.
5. **Pass@1 went down after fine-tuning.** Every fine-tuned model has lower average Pass@1 than the base model (≈0.28 vs 0.292), mostly from GPQA (0.299 → ≈0.25) and Minerva (0.309 → ≈0.28). The models find a correct answer more often across 8 attempts but are less consistent per attempt, a trade-off the leaderboard's pass@8 metric does not show.
6. **Caveat: differences are small.** AIME and CN Math have only 30 problems (one problem = 0.033), and we trained each configuration once. With a single run per configuration, differences of about 0.01 between fine-tuned models cannot be separated from run-to-run noise. Treat them as trends, not definitive rankings.

**What we would try next:** cluster-balanced sampling over **all** domains (no answer-format filter), and model-based difficulty (pass rate of the base model over several samples) instead of length as the difficulty proxy.

---

## Reproducing

### Environment
```bash
conda create -n myenv python=3.11 -y && conda activate myenv
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt -r requirements/deepspeed.txt -r requirements/vllm.txt
pip install word2number scikit-learn
pip install --no-deps latex2sympy2==1.9.1 sentence-transformers
hf auth login        # also request access to Idavidrein/gpqa
```
DeepSpeed needs the CUDA toolkit (`CUDA_HOME`). On a cloud A100 we installed `cuda-toolkit-12-8` to match PyTorch's CUDA 12.8. If an A100 reports **MIG enabled** with no MIG devices, PyTorch sees no GPU; disable it with `sudo nvidia-smi -i 0 -mig 0` and reboot.

### Data selection
```bash
python scripts/make_random.py          # -> LLaMA-Factory/data/random15k.json
python scripts/select_advanced.py      # -> LLaMA-Factory/data/advanced15k.json
python scripts/select_advanced_hard.py # -> advanced_hard15k.json (edit ROOT paths)
```
Register each file in `LLaMA-Factory/data/dataset_info.json`, e.g. `"random15k": {"file_name": "random15k.json"}`.

### Training
```bash
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train ../configs/random15k_1gpu.yaml
```
The advanced-hard config contains machine-specific absolute paths. Replace them with your own.

### Evaluation
```bash
cd Project-Reasoning-SFT-LLM/eval
MODEL=<path or HF repo> OUTPUT_DIR=./outputs_<run> bash eval_local.sh
grep -H -E "Acc:|Pass@1:" outputs_<run>/log_*.txt
```
Known issue: vLLM occasionally finishes a benchmark (scores written) but does not exit. If the GPU stays at 0% utilization for several minutes, kill the `eval.py` and `VLLM::EngineCore` processes. The loop continues with the next benchmark.

---

## Changes from the course README

The course instructions predate the current LLaMA-Factory release. Fixes we needed:

| Course README | What we did | Why |
|---|---|---|
| `git depth -1 clone ...` | `git clone --depth 1 ...` | Typo |
| Python 3.10 | Python 3.11 | LLaMA-Factory now requires Python ≥ 3.11 |
| `pip install -r requirements.txt` | not needed | File no longer exists in LLaMA-Factory |
| `pip install -e ".[torch,metrics]" --no-build-isolation` | `pip install -e .` + `requirements/metrics.txt` | Extras removed; `--no-build-isolation` fails with a missing `hatchling` backend |
| `pip install deepspeed==0.16.8`, `pip install vllm` | `requirements/deepspeed.txt`, `requirements/vllm.txt` | Versions compatible with LLaMA-Factory's PyTorch |
| `--account=ece_6514` | `--account=tml_2026` | This semester's ARC account |
| `ds_z3_offload_config.json` | `ds_z3_config.json` | No CPU offload needed on 80 GB GPUs; offload is much slower |
| SLURM 1-hour limit | ~22–26 h per run on one A100 | Full SFT on 15K long-reasoning examples |

**Compute:** ARC GPU queues were fully allocated during the project window (estimated start dates after the deadline), so training and evaluation ran on A100 80GB instances on a research cloud testbed and on a GH200 machine available to team members.
