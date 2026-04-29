# NER Project - 环境配置与运行说明

## 环境配置

```bash
# 创建conda环境（推荐Python 3.10+）
conda create -n ner python=3.10 -y
conda activate ner

# 安装PyTorch（CUDA 12.x，适用于RTX 4080）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install scikit-learn sklearn-crfsuite
```

## 运行

### 任务1：HMM（无需GPU）

```bash
python hmm_ner.py              # 运行中英文两个数据集
python hmm_ner.py English      # 只运行英文
python hmm_ner.py Chinese      # 只运行中文
```

### 任务2：CRF（无需GPU）

```bash
python crf_ner.py              # 运行中英文两个数据集
python crf_ner.py English
python crf_ner.py Chinese
```

### 任务3：Transformer+CRF（推荐GPU）

```bash
python transformer_crf_ner.py              # 运行中英文两个数据集（自动检测GPU）
python transformer_crf_ner.py English
python transformer_crf_ner.py Chinese
```

## 评测

```bash
# 以英文HMM结果为例
python -c "from check import check; check('English', 'English/validation.txt', 'hmm_result_english.txt')"

# 中文CRF结果
python -c "from check import check; check('Chinese', 'Chinese/validation.txt', 'crf_result_chinese.txt')"

# Transformer+CRF
python -c "from check import check; check('English', 'English/validation.txt', 'transformer_crf_result_english.txt')"
python -c "from check import check; check('Chinese', 'Chinese/validation.txt', 'transformer_crf_result_chinese.txt')"
```

## 面试测试

面试时提供 `test.txt` 后，将其放入对应语言目录（如 `English/test.txt`），重新运行对应脚本即可自动生成测试结果文件。

## 输出文件

| 任务 | 英文输出 | 中文输出 |
| ---- | ------- | ------- |
| HMM | `hmm_result_english.txt` | `hmm_result_chinese.txt` |
| CRF | `crf_result_english.txt` | `crf_result_chinese.txt` |
| Transformer+CRF | `transformer_crf_result_english.txt` | `transformer_crf_result_chinese.txt` |

## 一键查看所有模型评测对比

跑完任意几个模型后，运行：

```bash
python evaluate_all.py
```

会自动用 `check.py` 评测所有已生成的预测文件，并打印对比表。

## 当前实验结果（验证集 micro avg）

| 模型 | 数据集 | Precision | Recall | F1 |
| ---- | ------ | --------- | ------ | --- |
| HMM | English | 0.7152 | 0.7735 | **0.7432** |
| HMM | Chinese | 0.8664 | 0.8892 | **0.8776** |
| CRF | English | 0.9077 | 0.8677 | **0.8873** |
| CRF | Chinese | 0.9485 | 0.9539 | **0.9512** |
| Transformer+CRF | English | — | — | *待 4080 机器跑* |
| Transformer+CRF | Chinese | — | — | *待 4080 机器跑* |
