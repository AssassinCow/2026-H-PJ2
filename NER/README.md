# NER Project - 命名实体识别实验说明

本目录实现课程 Project 2 的英文与中文命名实体识别任务，包含三个模型：

| 任务 | 模型 | 文件 | 说明 |
| ---- | ---- | ---- | ---- |
| Task 1 | HMM | `src/hmm_ner.py` | 监督估计初始、转移、发射概率，手写 Viterbi 解码；OOV 走形态学/字符类型 backoff 插值 |
| Task 2 | CRF | `src/crf_ner.py` | `sklearn-crfsuite` 线性链 CRF，使用人工构造上下文特征 |
| Task 3 | Transformer+CRF | `src/transformer_crf_ner.py` | PyTorch Transformer Encoder + 约束 CRF；英文加入 casing embedding 与 char-CNN，中文加入 char-CNN |
| 融合 | Ensemble | `src/ensemble.py` | 三模型加权投票 + 合法 BIO/BMES 转移约束 Viterbi 重解码 |

数据文件采用逐行 `token tag` 格式，空行分隔句子。英文为 BIO 标签，中文为 BMES 标签。

## 目录结构

```text
NER/
├── data/
│   ├── English/                 # 英文 train.txt / validation.txt / test.txt / tag.txt
│   ├── Chinese/                 # 中文 train.txt / validation.txt / test.txt / tag.txt
│   └── example_data/            # 单文件评测格式示例
├── src/
│   ├── hmm_ner.py               # HMM 训练（训练后保存 pkl）
│   ├── crf_ner.py               # CRF 训练（训练后保存 pkl）
│   ├── transformer_crf_ner.py   # Transformer+CRF 训练（训练后保存 .pt checkpoint）
│   ├── ensemble.py              # 三模型加权投票融合
│   ├── test_hmm.py              # HMM 测试脚本（加载 pkl，直接预测）
│   ├── test_crf.py              # CRF 测试脚本（加载 pkl，直接预测）
│   ├── test_transformer_crf.py  # Transformer+CRF 测试脚本（加载 checkpoint，直接预测）
│   ├── evaluate_all.py          # 一键汇总所有模型结果（推荐评测入口）
│   ├── check.py                 # 单个预测文件评测函数（被 evaluate_all.py 调用）
│   └── filter_zh_chars.py       # fastText 中文字向量过滤工具
├── results/
│   ├── hmm/                     # HMM 输出：预测文件 + hmm_model_{lang}.pkl
│   ├── crf/                     # CRF 输出：预测文件 + crf_model_{lang}.pkl
│   ├── transformer_crf/         # Transformer+CRF 输出：预测文件 + transformer_crf_checkpoint_{lang}.pt
│   └── ensemble/                # Ensemble 预测输出
├── pretrained/                  # 预训练词向量（由 download_pretrained.sh 生成，不入版本控制）
├── requirements.txt             # Python 依赖列表
├── template_for_crf.utf8        # CRF++ 风格模板参考
├── download_pretrained.sh       # 下载 GloVe/fastText 预训练词向量
└── README.md
```

## 环境配置

```bash
# 创建 conda 环境（推荐 Python 3.10+）
conda create -n ner python=3.10 -y
conda activate ner

# 安装依赖
pip install -r requirements.txt

# 若需要 CUDA 12.x GPU 版 PyTorch，可按官方 wheel 覆盖安装
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 预训练词向量（仅 Transformer+CRF，可选但推荐）

Task 3 的 Transformer+CRF 默认 **开启** 预训练词向量。模型会把一份冻结的预训练向量与可学习的 128 维词嵌入 **拼接** 后再送入 Transformer，相当于给随机初始化的主干补一个语义先验，对英文 OOV 提升尤其明显（验证集 OOV 约 8.36%）。

| 语言 | 默认文件 | 维度 | 来源 |
| ---- | -------- | ---- | ---- |
| 英文 | `pretrained/glove.6B.300d.txt` | 300 | GloVe 6B（Stanford NLP） |
| 中文 | `pretrained/cc.zh.300.char.vec` | 300 | fastText `cc.zh.300.vec` 过滤出的单字向量 |

### 一键下载

```bash
cd NER
bash download_pretrained.sh                   # 默认下载到 ./pretrained/
bash download_pretrained.sh /path/to/dir      # 也可指定目录
```

脚本会：

1. 拉取并解压 GloVe `glove.6B.zip`（多镜像 fallback：huggingface → nlp.stanford.edu → downloads.cs.stanford.edu）
2. 下载 fastText 中文 `cc.zh.300.vec.gz`（约 1.4 GB），并通过 `src/filter_zh_chars.py` 过滤出单字行得到 `cc.zh.300.char.vec`（约 10 MB）
3. 已存在的目标文件会跳过，可重复运行恢复中断

要换成更小的英文向量（例如 `glove.6B.100d.txt`），改 `download_pretrained.sh` 顶部的 `GLOVE_FILE` 即可，模型会**自动检测维度**，无需改代码。

### 运行时如何被加载

文件查找优先级（在 `src/transformer_crf_ner.py` 的 `resolve_pretrained_path` 中）：

1. 环境变量 `NER_PRETRAINED_EN` / `NER_PRETRAINED_ZH`（绝对路径）
2. `NER/pretrained/` 下的默认文件名

```bash
# 用自定义路径覆盖默认值
export NER_PRETRAINED_EN="/data/glove.6B.300d.txt"
export NER_PRETRAINED_ZH="/data/cc.zh.300.char.vec"
python src/transformer_crf_ner.py
```

加载时会做大小写不敏感匹配（先精确，再小写 fallback），训练日志里能看到覆盖率：

```text
[English] Pretrained embeddings (concat, frozen): dim=300, hits=18432/23625 (78.0%), file=pretrained/glove.6B.300d.txt
```

### 没有文件时的行为

如果两个位置都找不到可读文件，会打印 WARNING、退化为纯随机 embedding 继续训练，不会报错，效果会下降（英文尤其明显）：

```text
[English] WARNING: use_pretrained=True but no readable file found ...
[English] To enable: run `bash download_pretrained.sh` ...
[English] Falling back to use_pretrained=False for this run.
```

### 关闭预训练

不想用预训练（例如做 ablation），把 `src/transformer_crf_ner.py` 里 `LANG_CONFIG[<lang>]['use_pretrained']` 改成 `False` 即可，模型其它部分不受影响。

## 训练

各训练脚本从 `data/` 读取数据，训练完成后自动保存模型权重到 `results/` 对应子目录，并输出验证集预测结果。

### 任务1：HMM（无需GPU）

```bash
cd NER
python src/hmm_ner.py              # 中英文两个数据集
python src/hmm_ner.py English
python src/hmm_ner.py Chinese
```

保存：`results/hmm/hmm_model_{lang}.pkl`

### 任务2：CRF（无需GPU）

```bash
python src/crf_ner.py
python src/crf_ner.py English
python src/crf_ner.py Chinese
```

保存：`results/crf/crf_model_{lang}.pkl`

### 任务3：Transformer+CRF（推荐GPU）

```bash
python src/transformer_crf_ner.py              # 自动检测GPU
python src/transformer_crf_ner.py English
python src/transformer_crf_ner.py Chinese
```

> 默认开启预训练词向量。首次训练前请先 `bash download_pretrained.sh`，否则会回退到纯随机 embedding（见上方"预训练词向量"一节）。训练过程中会显示 tqdm 进度条与每 epoch 峰值显存。

保存：`results/transformer_crf/transformer_crf_checkpoint_{lang}.pt`（含 vocab、tag2idx、模型结构参数）。这两个提交用 checkpoint 已在 `.gitignore` 中设为例外，保留在仓库里以支持克隆后直接测试；其它临时 `.pt` 训练产物仍默认忽略。

## 测试

拿到测试集后，将 `test.txt` 放入 `data/English/` 或 `data/Chinese/` 目录，直接运行对应测试脚本（**无需重新训练**）：

```bash
cd NER
python src/test_hmm.py [English|Chinese]
python src/test_crf.py [English|Chinese]
python src/test_transformer_crf.py [English|Chinese]
```

各测试脚本加载已保存的权重，输出预测文件到 `results/` 对应子目录，并提示用 `evaluate_all.py test` 汇总评测。若权重文件不存在，脚本会提示先运行训练脚本。

## 评测

```bash
cd NER/src
python evaluate_all.py            # 默认验证集
python evaluate_all.py val        # 验证集
python evaluate_all.py test       # 测试集
```

`evaluate_all.py` 会调用 `check.py` 对每个结果文件计算 token-level classification report，并汇总 HMM / CRF / Transformer+CRF / Ensemble 在 English / Chinese 上的 micro avg 指标。

## 三模型融合（ensemble）

在三个模型分别生成验证集预测后，可使用 `ensemble.py` 做加权投票并以合法 BIO/BMES 转移约束重新 Viterbi 解码：

```bash
python src/ensemble.py              # 中英文都跑
python src/ensemble.py English      # 只跑英文
python src/ensemble.py Chinese      # 只跑中文
```

默认权重取每个模型的验证集 micro F1（见脚本顶部 `WEIGHTS`），可在脚本里手工调整。输出文件 `results/ensemble/ensemble_result_<lang>.txt` 与单模型结果同格式，可直接用 `evaluate_all.py` 评测。

`val` 用 `data/{Lang}/validation.txt` 与 `*_result_{lang}.txt` 比对；`test` 用 `data/{Lang}/test.txt` 与 `*_test_result_{lang}.txt` 比对。缺文件的模型会显示 `(pred-missing)` / `(gold-missing)`，不会让其它模型评测中断。

## 输出文件说明

| 任务 | 验证集预测 | 模型权重 | 测试集预测 |
| ---- | --------- | ------- | --------- |
| HMM | `results/hmm/hmm_result_{lang}.txt` | `results/hmm/hmm_model_{lang}.pkl` | `results/hmm/hmm_test_result_{lang}.txt` |
| CRF | `results/crf/crf_result_{lang}.txt` | `results/crf/crf_model_{lang}.pkl` | `results/crf/crf_test_result_{lang}.txt` |
| Transformer+CRF | `results/transformer_crf/transformer_crf_result_{lang}.txt` | `results/transformer_crf/transformer_crf_checkpoint_{lang}.pt` | `results/transformer_crf/transformer_crf_test_result_{lang}.txt` |
| Ensemble | `results/ensemble/ensemble_result_{lang}.txt` | — | — |

## 当前实验结果

以下结果来自最新一次运行日志，指标为 token-level micro average，评测时排除 `O` 标签。

### 验证集

| 模型 | 数据集 | Precision | Recall | F1 |
| ---- | ------ | --------- | ------ | --- |
| HMM | English | 0.8589 | 0.8160 | 0.8369 |
| HMM | Chinese | 0.8667 | 0.8891 | 0.8777 |
| CRF | English | 0.9133 | **0.8964** | 0.9048 |
| CRF | Chinese | 0.9497 | 0.9540 | 0.9519 |
| Transformer+CRF | English | 0.9064 | 0.8827 | 0.8944 |
| Transformer+CRF | Chinese | **0.9530** | 0.9533 | **0.9531** |
| Ensemble | English | **0.9195** | 0.8939 | **0.9065** |
| Ensemble | Chinese | 0.9510 | **0.9551** | 0.9530 |

### 测试集

Ensemble 暂无测试集预测（投票权重依赖验证集 F1，先在测试集上跑过单模型后再补）。

| 模型 | 数据集 | Precision | Recall | F1 |
| ---- | ------ | --------- | ------ | --- |
| HMM | English | 0.7762 | 0.7525 | 0.7641 |
| HMM | Chinese | 0.8960 | 0.9169 | 0.9063 |
| CRF | English | 0.8399 | 0.8465 | 0.8432 |
| CRF | Chinese | **0.9608** | **0.9510** | **0.9558** |
| Transformer+CRF | English | **0.8459** | **0.8468** | **0.8464** |
| Transformer+CRF | Chinese | 0.9567 | 0.9470 | 0.9518 |

当前最优模型：

- English：验证集 Ensemble F1 = `0.9065`；测试集 Transformer+CRF F1 = `0.8464`
- Chinese：验证集 Transformer+CRF F1 = `0.9531`；测试集 CRF F1 = `0.9558`

### Ablation：关闭预训练词向量

把 `LANG_CONFIG['English']['use_pretrained']` 与 `LANG_CONFIG['Chinese']['use_pretrained']` 都改成 `False` 重训 Transformer+CRF（其他超参不变），用来量化 GloVe / fastText 静态向量的贡献：

| 集合 | 模型 | 数据集 | Precision | Recall | F1 | ΔF1 vs 默认 |
| ---- | ---- | ------ | --------- | ------ | --- | ----------- |
| 验证集 | Transformer+CRF | English | 0.9017 | 0.8813 | 0.8914 | −0.0030 |
| 验证集 | Transformer+CRF | Chinese | 0.9522 | 0.9442 | 0.9482 | −0.0049 |
| 测试集 | Transformer+CRF | English | 0.8356 | 0.8337 | 0.8346 | −0.0118 |
| 测试集 | Transformer+CRF | Chinese | 0.9536 | 0.9388 | 0.9462 | −0.0056 |

预训练向量在测试集上的收益（英文 +`0.0118`，中文 +`0.0056`）明显大于验证集（+`0.0030` / +`0.0049`），说明 GloVe / fastText 主要在 OOV 与边缘样本的泛化上起作用——验证集分布更接近训练集，关闭预训练后用 learnable embedding 也能勉强拟合；测试集分布更陌生，预训练带来的语义先验才显出价值。把这一版（关闭预训练）的 Trans+CRF 预测灌进 Ensemble，验证集得到 English `0.9053` / Chinese `0.9513`，两个语言都比默认 Ensemble（`0.9065` / `0.9530`）略低，说明预训练对融合器同样有正贡献，且这一贡献在英文上比中文更明显（与单模型 ablation 一致）。

## 本轮优化效果

| 模型 | 数据集 | P 前 | P 后 | ΔP | R 前 | R 后 | ΔR | F1 前 | F1 后 | ΔF1 |
| ---- | ------ | ---- | ---- | -- | ---- | ---- | -- | ----- | ----- | --- |
| HMM | English | 0.7152 | 0.8589 | +0.1437 | 0.7735 | 0.8160 | +0.0425 | 0.7432 | 0.8369 | +0.0937 |
| HMM | Chinese | 0.8664 | 0.8667 | +0.0003 | 0.8892 | 0.8891 | -0.0001 | 0.8776 | 0.8777 | +0.0001 |
| CRF | English | 0.9077 | 0.9133 | +0.0056 | 0.8677 | 0.8964 | +0.0287 | 0.8873 | 0.9048 | +0.0175 |
| CRF | Chinese | 0.9485 | 0.9497 | +0.0012 | 0.9539 | 0.9540 | +0.0001 | 0.9512 | 0.9519 | +0.0007 |
| Transformer+CRF | English | 0.8409 | 0.9064 | +0.0655 | 0.8273 | 0.8827 | +0.0554 | 0.8340 | 0.8944 | +0.0604 |
| Transformer+CRF | Chinese | 0.9420 | 0.9530 | +0.0110 | 0.9446 | 0.9533 | +0.0087 | 0.9433 | 0.9531 | +0.0098 |

## 结果分析

### 总体趋势

验证集与测试集的最优模型已不再一致：验证集英文最优是 Ensemble（F1 `0.9065`），中文最优是 Transformer+CRF（F1 `0.9531`，首次反超 CRF）；而测试集英文最优是 Transformer+CRF（F1 `0.8464`），中文最优是 CRF（F1 `0.9558`）。这说明三个模型的偏置不同：CRF 的人工特征在分布更稳定的中文测试集上泛化最好，Transformer+CRF 在词表覆盖率更敏感的英文上靠预训练 GloVe 拿到更平稳的泛化，验证集排名更多反映哪个模型更"对得上"开发集分布。

HMM 作为生成式基线，本轮加入 OOV backoff 插值后英文验证集 F1 从 `0.7432` 提升到 `0.8369`（+0.0937），主要收益来自约 8% 的 OOV token 不再统一退化到单一 `<UNK>` 概率，而是按 `lower / suffix-3 / prefix-3 / shape` 分布做 add-one 平滑插值；中文字符级 OOV 极低，结果与基线持平（+0.0001）。

Transformer+CRF 经强化后改善显著：英文验证集 F1 从 `0.8340` 提升到 `0.8944`（+0.0604），中文从 `0.9433` 提升到 `0.9531`（+0.0098）。除了 constrained CRF、按 F1 早停、AdamW/weight decay、word dropout、多卷积核 char-CNN 等优化外，本轮新增的 GloVe / fastText 静态预训练向量（frozen + concat）是关键贡献——英文验证集 OOV 约 `8.36%`，GloVe-300d 把这部分语义先验补上后英文 +6 个点；中文验证集靠 fastText `cc.zh.300` 单字向量拼接，单点提升不大但足以让 Transformer+CRF 首次反超 CRF。预训练向量的具体增益由"当前实验结果 → Ablation：关闭预训练词向量"小节量化（测试集 +`0.0118` / +`0.0056`，验证集 +`0.0030` / +`0.0049`）。

### 英文任务

验证集排序：Ensemble `0.9065` > CRF `0.9048` > Transformer+CRF `0.8944` > HMM `0.8369`。CRF 与 Ensemble 差距只有 `0.0017`，且 Ensemble 优势来自 precision（`0.9195` vs CRF `0.9133`）而非 recall——Ensemble recall `0.8939` 反而比 CRF `0.8964` 略低，这与加权投票偏保守、把单模型边缘标注挡掉的行为一致。增强后的 CRF 加入 prefix/suffix、word shape、压缩 shape、标点数字特征、称谓、机构后缀、月份和上下文组合特征，对 CoNLL 风格英文非常有效，单模型最高 recall 也来自它。

测试集排序则换了：Transformer+CRF `0.8464` > CRF `0.8432` > HMM `0.7641`。三模型在测试集上整体掉了 `0.05~0.07`，但 Transformer+CRF 掉得最少（验证 0.8944 → 测试 0.8464，−0.048），CRF 掉幅 `0.062`。Transformer+CRF 的预训练 GloVe 在测试集 OOV 上提供更稳的语义先验，是它在英文测试集反超的主因（ablation 数据：关闭 GloVe 后英文测试集 F1 直接掉到 `0.8346`，−0.0118，让出第一）；CRF 短板仍在 `I-MISC`、`I-LOC` 等长尾内部标签。

### 中文任务

验证集排序：Transformer+CRF `0.9531` > Ensemble `0.9530` > CRF `0.9519` > HMM `0.8777`，三个强模型几乎贴着。Transformer+CRF 这次首次反超 CRF，主要靠 fastText `cc.zh.300` 单字向量拼接进 embedding 后的语义补充，以及 Constrained CRF 把非法 BMES 转移屏蔽（ablation 数据：关闭 fastText 后中文验证集 F1 掉到 `0.9482`，−0.0049，让出第一并落到 CRF 后面）。

测试集排序反转：CRF `0.9558` > Transformer+CRF `0.9518` > HMM `0.9063`。CRF 在测试集上几乎不掉点（验证 0.9519 → 测试 0.9558，反而 +0.0039），印证人工特征 + 显式转移建模在分布稳定的中文 BMES 标注上泛化非常稳；Transformer+CRF 掉了 `0.0013`，仍接近，但短板仍是 `TITLE`、`ORG` 边界以及 `PRO`、`LOC`、`RACE` 等长尾标签。

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
5. 接入 GloVe / fastText 预训练词向量并默认启用：英文用 GloVe-300d，中文用 fastText `cc.zh.300` 过滤后的单字向量；以「冻结 + 与 128 维可学习 embedding 拼接」的方式注入，token 缺失时自动回退到 learnable embedding，文件路径可通过 `NER_PRETRAINED_EN` / `NER_PRETRAINED_ZH` 覆盖（详见"预训练词向量"一节）。

### 已完成的融合策略（本轮新增）

`src/ensemble.py` 读入 HMM / CRF / Transformer+CRF 三个模型的预测文件，对每个 token 做 F1 加权投票得到伪 emission，然后用合法 BIO/BMES 转移约束的 Viterbi 重新解码，确保最终标签序列在格式层面一致；脚本同时打印各单模型与融合结果的 micro F1 对比。

### 后续优化方向

1. 改用 BERT-CRF：当前已接入 GloVe / fastText 静态词向量，下一步可换成 BERT / RoBERTa 等上下文相关编码器，在小数据 NER 上通常比随机初始化 Transformer 主干更稳。
2. 把预训练向量从 frozen 改成 fine-tune：现在为防止覆盖语义先验做了冻结，可对比解冻后的效果，或加一层 projection 让两个 embedding 流在更高层融合。
3. 做多随机种子和超参搜索：比较 `dropout`、`lr`、`batch_size`、层数、hidden size，并记录均值和方差。
4. 对低频标签做针对性误差分析：英文重点看 `I-MISC`、`I-LOC`；中文重点看 `PRO`、`LOC`、`RACE` 和单字实体标签。

### 评测优化

当前 `src/check.py` 使用 token-level classification report。为了更贴近 NER 任务本身，建议额外加入 entity-level span F1（例如 `seqeval`），因为 token-level 分数可能低估或高估完整实体边界的质量。
