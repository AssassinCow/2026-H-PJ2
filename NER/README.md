# NER Project - 命名实体识别实验说明

本目录实现课程 Project 2 的英文与中文命名实体识别任务，包含三个模型：

| 任务 | 模型 | 文件 | 说明 |
| ---- | ---- | ---- | ---- |
| Task 1 | HMM | `hmm_ner.py` | 监督估计初始、转移、发射概率，手写 Viterbi 解码；OOV 走形态学/字符类型 backoff 插值 |
| Task 2 | CRF | `crf_ner.py` | `sklearn-crfsuite` 线性链 CRF，使用人工构造上下文特征 |
| Task 3 | Transformer+CRF | `transformer_crf_ner.py` | PyTorch Transformer Encoder + 约束 CRF；英文加入 casing embedding 与 char-CNN，中文加入 char-CNN |
| 融合 | Ensemble | `ensemble.py` | 三模型加权投票 + 合法 BIO/BMES 转移约束 Viterbi 重解码 |

数据文件采用逐行 `token tag` 格式，空行分隔句子。英文为 BIO 标签，中文为 BMES 标签。

## 目录结构

```text
NER/
├── English/                         # 英文 train.txt / validation.txt / tag.txt
├── Chinese/                         # 中文 train.txt / validation.txt / tag.txt
├── example_data/                    # check.py 的格式示例
├── hmm_ner.py                       # HMM 基线（OOV backoff 插值）
├── crf_ner.py                       # CRF 模型
├── transformer_crf_ner.py           # Transformer+CRF 模型
├── ensemble.py                      # 三模型加权投票融合
├── check.py                         # 单个预测文件评测
├── evaluate_all.py                  # 一键汇总所有模型结果
├── template_for_crf.utf8            # CRF++ 风格模板参考
└── *_result_*.txt                   # 已生成的验证集预测结果
```

## 环境配置

```bash
# 创建 conda 环境（推荐 Python 3.10+）
conda create -n ner python=3.10 -y
conda activate ner

# 安装 PyTorch（CUDA 12.x，适用于 RTX 4080；CPU 环境可改用官方 CPU wheel）
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

脚本默认读取对应语言的 `train.txt` 与 `validation.txt`，并把验证集预测写入当前目录下的结果文件。

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

或者使用一键汇总：

```bash
python evaluate_all.py
```

## 三模型融合（ensemble）

在三个模型分别生成验证集预测后，可使用 `ensemble.py` 做加权投票并以合法 BIO/BMES 转移约束重新 Viterbi 解码：

```bash
python ensemble.py              # 中英文都跑
python ensemble.py English      # 只跑英文
python ensemble.py Chinese      # 只跑中文
```

默认权重取每个模型的验证集 micro F1（见脚本顶部 `WEIGHTS`），可在脚本里手工调整。输出文件 `ensemble_result_<lang>.txt` 与单模型结果同格式，可直接喂给 `check.py` 评测。

## 输出文件说明

| 任务 | 英文输出 | 中文输出 |
| ---- | ------- | ------- |
| HMM | `hmm_result_english.txt` | `hmm_result_chinese.txt` |
| CRF | `crf_result_english.txt` | `crf_result_chinese.txt` |
| Transformer+CRF | `transformer_crf_result_english.txt` | `transformer_crf_result_chinese.txt` |
| Ensemble | `ensemble_result_english.txt` | `ensemble_result_chinese.txt` |

若测试阶段提供 `test.txt`，将其放入 `English/` 或 `Chinese/` 目录后重新运行对应脚本，会额外生成 `*_test_result_*.txt`。

## 当前实验结果

以下结果来自最新一次运行日志，指标为验证集 token-level micro average，评测时排除 `O` 标签。

| 模型 | 数据集 | Precision | Recall | F1 |
| ---- | ------ | --------- | ------ | --- |
| HMM | English | 0.8589 | 0.8160 | 0.8369 |
| HMM | Chinese | 0.8667 | 0.8891 | 0.8777 |
| CRF | English | **0.9133** | **0.8964** | **0.9048** |
| CRF | Chinese | **0.9497** | **0.9540** | **0.9519** |
| Transformer+CRF | English | 0.9013 | 0.8774 | 0.8892 |
| Transformer+CRF | Chinese | 0.9475 | 0.9430 | 0.9452 |

当前最优模型：

- English：CRF，F1 = `0.9048`
- Chinese：CRF，F1 = `0.9519`

## 本轮优化效果

| 模型 | 数据集 | P 前 | P 后 | ΔP | R 前 | R 后 | ΔR | F1 前 | F1 后 | ΔF1 |
| ---- | ------ | ---- | ---- | -- | ---- | ---- | -- | ----- | ----- | --- |
| HMM | English | 0.7152 | 0.8589 | +0.1437 | 0.7735 | 0.8160 | +0.0425 | 0.7432 | 0.8369 | +0.0937 |
| HMM | Chinese | 0.8664 | 0.8667 | +0.0003 | 0.8892 | 0.8891 | -0.0001 | 0.8776 | 0.8777 | +0.0001 |
| CRF | English | 0.9077 | 0.9133 | +0.0056 | 0.8677 | 0.8964 | +0.0287 | 0.8873 | 0.9048 | +0.0175 |
| CRF | Chinese | 0.9485 | 0.9497 | +0.0012 | 0.9539 | 0.9540 | +0.0001 | 0.9512 | 0.9519 | +0.0007 |
| Transformer+CRF | English | 0.8409 | 0.9013 | +0.0604 | 0.8273 | 0.8774 | +0.0501 | 0.8340 | 0.8892 | +0.0552 |
| Transformer+CRF | Chinese | 0.9420 | 0.9475 | +0.0055 | 0.9446 | 0.9430 | -0.0016 | 0.9433 | 0.9452 | +0.0019 |

## 结果分析

### 总体趋势

CRF 在中英文上都取得最高 F1，说明在当前数据规模下，强人工特征和显式转移建模仍然最稳定。HMM 作为生成式基线，本轮加入 OOV backoff 插值后英文 F1 从 `0.7432` 提升到 `0.8369`（+0.0937），主要收益来自验证集约 8% 的 OOV token 不再统一退化到单一 `<UNK>` 概率，而是按 `lower / suffix-3 / prefix-3 / shape` 分布做 add-one 平滑插值；中文数据字符级 OOV 极低，所以 backoff 几乎不起作用，结果与基线持平（+0.0001）。

Transformer+CRF 经强化后明显改善，尤其英文 F1 从 `0.8340` 提升到 `0.8892`。提升主要来自 constrained CRF 合法转移约束、按验证集 F1 早停、AdamW/weight decay、word dropout、embedding dropout 和多卷积核 char-CNN。它仍未超过 CRF，主要原因是词向量与 Transformer 主干仍从零训练，缺少预训练语义知识；英文验证集 OOV 约 `8.36%`，对随机初始化词表模型仍不友好。

### 英文任务

CRF 的英文 F1 为 `0.9048`，高于 Transformer+CRF 的 `0.8892`。增强后的 CRF 加入了 prefix/suffix、word shape、压缩 shape、标点数字特征、人物称谓、机构后缀、月份和更丰富的上下文组合，因此对 CoNLL 风格英文实体非常有效。Transformer+CRF 的英文结果已经接近 CRF，但 `I-MISC`、`I-LOC` 等低频内部标签仍是主要短板。

### 中文任务

中文 CRF F1 为 `0.9519`，Transformer+CRF 为 `0.9452`，差距较小。中文数据是字符级标注，词表规模小、OOV 低，模型更容易记住局部字符模式。CRF 的窗口、bigram/trigram 字符特征和机构/职位/学历/地名后缀特征贴合中文 BMES 边界识别，因此当前仍略优。错误主要集中在 `TITLE`、`ORG` 边界，以及样本很少的 `PRO`、`LOC`、`RACE`、单字实体标签上。

## 模型优化方向

### 已完成的 HMM 优化（本轮新增）

1. 用形态学/字符类型 backoff 替代单一 `<UNK>` 概率：
   - 英文：当 token 不在训练词表时，用 `lower / 3-char suffix / 3-char prefix / 字形 shape` 四个特征的 add-one 平滑分布做线性插值。
   - 中文：未登录字符时使用粗粒度 `char_type`（CJK / DIGIT / ALPHA / PUNCT / OTHER）backoff。
2. 实现层面保持纯手写、零 ML 框架依赖；在词表内 token 上的发射概率与基线完全一致，因此最坏情况退回基线，最好情况显著改善 OOV 上的标签判断。

### 已完成的 CRF 优化

1. 英文加入 prefix/suffix 1-4 位、word shape、压缩 shape、标点/数字/大小写、称谓、机构后缀、月份和上下文组合特征。
2. 中文加入字符类型、`[-3,+3]` 窗口、bigram/trigram 字符组合、组织机构后缀、职位后缀、学历关键词和地名后缀。
3. 调整 CRF 正则和迭代轮数：英文 `c1=0.05,c2=0.02,max_iterations=150`，中文 `c1=0.05,c2=0.05,max_iterations=150`。
4. 加入 BIO/BMES 序列修复后处理，降低非法标签转移造成的边界错误。

### 已完成的 Transformer+CRF 优化

1. 加入 constrained CRF，在 forward algorithm 和 Viterbi 解码中屏蔽非法 BIO/BMES 转移。
2. 将早停和最佳 checkpoint 选择指标从验证 loss 改为验证集 micro F1。
3. 使用 AdamW、weight decay、embedding dropout、word dropout 和固定随机种子。
4. 将 char-CNN 改为多卷积核结构，并在中文任务中启用 char-CNN。
5. 预留 GloVe/fastText 风格预训练词向量入口，可通过 `NER_PRETRAINED_EN` / `NER_PRETRAINED_ZH` 环境变量指定。

### 已完成的融合策略（本轮新增）

`ensemble.py` 读入 HMM / CRF / Transformer+CRF 三个模型的预测文件，对每个 token 做 F1 加权投票得到伪 emission，然后用合法 BIO/BMES 转移约束的 Viterbi 重新解码，确保最终标签序列在格式层面一致；脚本同时打印各单模型与融合结果的 micro F1 对比。

### 后续优化方向

1. 使用预训练表示：英文可接入 GloVe/fastText 或 BERT，中文可接入 Chinese BERT/RoBERTa。当前 Transformer 从零训练，是英文表现落后的主要原因。
2. 改用 BERT-CRF：在小数据 NER 上，预训练编码器 + CRF 通常比随机初始化 Transformer 更可靠。
3. 做多随机种子和超参搜索：比较 `dropout`、`lr`、`batch_size`、层数、hidden size，并记录均值和方差。
4. 对低频标签做针对性误差分析：英文重点看 `I-MISC`、`I-LOC`；中文重点看 `PRO`、`LOC`、`RACE` 和单字实体标签。

### 评测优化

当前 `check.py` 使用 token-level classification report。为了更贴近 NER 任务本身，建议额外加入 entity-level span F1（例如 `seqeval`），因为 token-level 分数可能低估或高估完整实体边界的质量。
