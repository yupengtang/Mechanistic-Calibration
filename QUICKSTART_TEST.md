# Quick Start: Test Experiment

快速测试实验框架是否可以正常运行。

---

## 前置检查

```bash
# 1. 检查 Python 依赖
python verify_setup.py

# 2. 确保 .env 文件存在（API 模型需要）
cat .env
# 应该包含: OPENROUTER_API_KEY=your-key-here
```

---

## 测试 1: 数据生成（候选池构建）

```bash
# 测试构建候选池（使用小样本）
python scripts/collect_candidates.py \
  --output data/test_candidate_pool.jsonl \
  --max-per-dataset 50 \
  --test-mode

# 检查输出
wc -l data/test_candidate_pool.jsonl
head -3 data/test_candidate_pool.jsonl
```

**预期结果：** 生成 JSONL 文件，包含 PubMedQA/SciFact/ContractNLI 候选问题

---

## 测试 2: BEAT-120 选择（边界过滤）

```bash
# 测试问题筛选（使用 API 模型，避免下载大模型）
python -m src.beat120_builder \
  --input data/test_candidate_pool.jsonl \
  --output data/test_beat120.jsonl \
  --target-size 10 \
  --models "openai/gpt-4o-mini" \
  --provider openrouter

# 检查输出
wc -l data/test_beat120.jsonl
```

**预期结果：** 筛选出 10 个在认知边界的问题

---

## 测试 3: 运行实验（最小配置）

```bash
# 使用测试模式（1 个问题，2 个条件，1 次重复）
python run_experiment.py \
  --questions data/test_beat120.jsonl \
  --output results/test_trials.jsonl \
  --test-mode \
  --device cpu

# 或使用提供的样本（如果存在）
python run_experiment.py \
  --questions frozen_artifacts/beat120_questions_SAMPLE.jsonl \
  --output results/test_trials.jsonl \
  --test-mode \
  --device cpu
```

**测试模式配置（自动应用）：**
- 温度: T=0.0 only
- 条件: A0, A1 + B0 + C1（2 个条件）
- 重复: R=1
- 问题: 只取第 1 个

**预期输出：**
```
Loading resources...
Loaded 1 questions
Running 1 model(s)
Testing 2 condition(s)

STARTING EXPERIMENT WITH PROPER PASS 1 SHARING
===============================================

Total Pass 1 runs: 1
Total Pass 2 runs: 2
Total trials: 2

Running trials: 100%|████████| 2/2 [00:XX<00:00]

EXPERIMENT COMPLETE
===============================================
Pass 1 runs:
  Success: 1
  Invalid: 0
Trials (Pass 2):
  Success: 2
  Errors: 0
Output: results/test_trials.jsonl
```

---

## 测试 4: 分析结果

```bash
# 测试分析脚本
python analyze_results.py \
  --input results/test_trials.jsonl \
  --output results/test_analysis/

# 检查输出
ls -lh results/test_analysis/
```

**预期结果：** 生成 CSV 文件（drift_adjusted.csv, ssi_behavioral.csv, taxonomy.csv 等）

---

## 测试 5: 生成图表

```bash
# 测试绘图脚本
python generate_figures.py \
  --analysis-dir results/test_analysis/ \
  --output-dir figures/test/

# 检查输出
ls -lh figures/test/
```

**预期结果：** 生成 PNG 图表文件

---

## 常见问题排查

### 问题 1: `ModuleNotFoundError: No module named 'torch'`

如果只想测试 API 模型（不需要本地模型）：

```bash
# 修改 config.json
# "run_mechanistic_core": false,
# "run_behavioral_anchors": true
```

### 问题 2: `ModuleNotFoundError: No module named 'dotenv'`

```bash
pip install python-dotenv
```

### 问题 3: OpenRouter API 错误

检查 `.env` 文件和 API key:
```bash
cat .env
# 确保格式正确: OPENROUTER_API_KEY=sk-...
```

### 问题 4: CUDA 不可用

使用 CPU 运行（本地模型会很慢）：
```bash
python run_experiment.py ... --device cpu
```

---

## 完整测试（不下载大模型）

如果你只想验证框架可以跑通，不想下载 70B/405B 模型：

```bash
# 1. 使用 API 模型运行完整实验（小规模）
python run_experiment.py \
  --questions frozen_artifacts/beat120_questions_SAMPLE.jsonl \
  --output results/api_test_trials.jsonl \
  --device cpu

# 2. 修改 config.json，只启用 API 模型
# "run_mechanistic_core": false,
# "run_behavioral_anchors": true

# 3. 运行分析
python analyze_results.py \
  --input results/api_test_trials.jsonl \
  --output results/api_test_analysis/
```

---

## 完整实验（需要 GPU 和大量时间）

```bash
# 使用完整 harness 运行所有模型
bash run_full_pipeline.sh
```

**预计时间（A100 GPU）：**
- 数据生成: ~30 min
- BEAT-120 选择: ~2 hours
- 核心实验（4 模型 × 120 问题 × 32 条件 × 3 重复）: ~20-40 hours
- 分析 + 图表: ~1 hour

---

## 成功标志

实验可以正常运行的标志：

✓ `verify_setup.py` 通过所有检查  
✓ `collect_candidates.py` 生成候选池  
✓ `beat120_builder.py` 筛选出边界问题  
✓ `run_experiment.py --test-mode` 完成 Pass 1 和 Pass 2  
✓ `analyze_results.py` 生成分析结果  
✓ `generate_figures.py` 生成图表

现在可以开始正式实验！
