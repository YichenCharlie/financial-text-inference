# Execution Guide

Run all commands from the repository root. GPU experiments require a
CUDA-enabled environment and the trained BERT checkpoint.

## 1. Environment

Recorded experiment environment:

| Component | Version / hardware |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.57.1 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |

`requirements.txt` is a package snapshot exported from the working
environment, not a minimal dependency specification.

For a new environment, install the experiment's PyTorch build first:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Package availability and the NVIDIA driver must support the selected CUDA
build. The package snapshot does not install the GPU driver or reproduce
the complete operating-system environment. Clean-environment reproduction
has not yet been verified.

Check the active environment before running GPU experiments:

```bash
python -c "import sys, torch, transformers; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 2. Data Preparation

Obtain the dataset from [FinChina-SA](https://github.com/YerayL/FinChina-SA)
and place the source files at:

- `data/raw/train.json`
- `data/raw/test.json`

Prepare the grouped training/validation split:

```bash
python practice/split_data.py
```

Expected processed files:

- `data/processed/train_samples.jsonl`: 17,174 samples
- `data/processed/validation_samples.jsonl`: 1,942 samples

The author-provided test set is reserved and is not evaluated by the
validation commands below.

## 3. Baseline and BERT Training

Train the TF-IDF baseline and generate its figures:

```bash
python scripts/stage1_train_baseline.py
python scripts/stage1_plot_results.py
```

Fine-tune BERT and export validation predictions:

```bash
python scripts/stage3_train_bert.py
python scripts/stage3_predict_validation.py
python scripts/stage3_plot_results.py
```

Before training, check the pretrained-model path configured in
`stage3_train_bert.py` and prepare the corresponding model files.

Subsequent BERT experiments expect the trained model, configuration, and
tokenizer at `models/stage3_bert_full/`. Inference scripts load these files
locally and do not download missing checkpoints.

## 4. Forward Verification

Compare single-sample and batch inference, verify the reconstructed
Encoder layer, and check the complete forward path:

```bash
python scripts/stage4_inference.py
python scripts/stage4_verify_bert_layer.py
python scripts/stage4_verify_bert_forward.py
```

Generate the architecture diagram:

```bash
python scripts/stage4_plot_bert_architecture.py
```

The layer and full-forward verification scripts use CPU execution to
check numerical agreement. They are not GPU speed benchmarks.

## 5. GPU Profiling

Record operator activity and module annotations:

```bash
python scripts/stage5_profile_bert.py
```

The script prints the timestamped trace and figure directories.
Analyze the latest generated trace:

```bash
TRACE_PATH=$(python - <<'PY'
from pathlib import Path

paths = sorted(Path("results/stage5_profiling").glob("*/trace.json"))
if not paths:
    raise SystemExit("No trace found. Run stage5_profile_bert.py first.")
print(paths[-1])
PY
)

python scripts/stage5_analyze_trace.py \
  --trace "$TRACE_PATH" \
  --forward 2 \
  --layer 2
```

Compare execution with and without instrumentation:

```bash
python scripts/stage5_measure_overhead.py
```

Profiler traces explain execution structure and attributed computation
time. Use separate unprofiled measurements for performance claims.

## 6. Mixed-Precision Evaluation

Inspect module-boundary dtypes and single-sample output differences:

```bash
python scripts/stage6_inspect_precision.py
```

Measure FP32 and AMP FP16 forward performance without hooks or Profiler:

```bash
python scripts/stage6_benchmark_precision.py
```

Compare predictions and metrics over the complete validation set:

```bash
python scripts/stage6_validate_precision.py
```

Profile both precision modes and generate the kernel comparison figure:

```bash
python scripts/stage6_profile_precision.py
```

The speed benchmark uses batch size 1 and a 512-token input. The full
validation comparison uses batch size 16. Each AMP forward enters a
separate autocast context.

## 7. Main Outputs

| Output | Path |
|---|---|
| BERT validation predictions | `results/stage3_bert_validation_predictions.jsonl` |
| Architecture diagram | `figures/stage4_bert/bert_architecture.png` |
| FP32/AMP timing results | `results/stage6_precision/benchmark.json` |
| Validation metrics and agreement | `results/stage6_precision/validation_summary.json` |
| Per-sample precision comparison | `results/stage6_precision/validation_comparison.jsonl` |
| Kernel comparison data | `results/stage6_precision/profiling/kernel_comparison.json` |
| Kernel comparison figure | `figures/stage6_precision/precision_kernel_comparison.png` |

Stage 5 uses timestamped output directories. Stage 6 uses fixed paths;
rerunning a Stage 6 script replaces its corresponding outputs. Preserve
previous results before rerunning if they are needed for comparison.

Run benchmarks without competing GPU workloads. Keep the model, input,
precision settings, and autocast scope consistent when comparing results.