# 人工智能(H) Project 2 实验报告：命名实体识别

课程：人工智能(H) 26春  
姓名：刘卓鑫  
学号：24300240170  
任务：在英文 CoNLL-2003 风格数据集与中文简历 NER 数据集上实现并比较 HMM、CRF、Transformer+CRF 三类序列标注模型。

## 1. 任务概述

命名实体识别（Named Entity Recognition, NER）要求对句子中的每个 token 预测实体边界和实体类别。本次实验中，我分别实现了三类序列标注模型，并在中文和英文数据集上比较它们的表现。项目包含三个主任务：

1. 手写 HMM 模型完成 NER，不使用机器学习框架。
2. 使用 CRF 完成 NER，能够在中英文数据集上训练、验证并输出预测文件。
3. 使用 Transformer+CRF 完成 NER，其中 Transformer 部分使用 PyTorch，CRF 层手写实现。

数据文件均采用逐行 `token tag` 格式，空行分隔句子。英文标签采用 BIO 格式，共 9 类标签；中文标签采用 BMES 格式，共 33 类标签。评测采用 `check.py` 输出的 token-level micro average F1，评测时排除 `O` 标签。后文中的结果都按这一标准统计。

## 2. 代码结构

我将主要代码都放在 `NER/` 目录下，训练脚本、测试脚本和结果文件分开存放，便于面试时直接复现：

```text
NER/
├── data/
│   ├── English/                 # 英文 train / validation / tag 文件
│   ├── Chinese/                 # 中文 train / validation / tag 文件
│   └── example_data/            # 评测格式示例
├── src/
│   ├── hmm_ner.py               # 任务一：手写 HMM
│   ├── crf_ner.py               # 任务二：sklearn-crfsuite CRF
│   ├── transformer_crf_ner.py   # 任务三：PyTorch Transformer + 手写 CRF
│   ├── ensemble.py              # 三模型融合
│   ├── test_hmm.py              # HMM 测试集预测
│   ├── test_crf.py              # CRF 测试集预测
│   ├── test_transformer_crf.py  # Transformer+CRF 测试集预测
│   ├── evaluate_all.py          # 汇总评测入口
│   ├── check.py                 # 单文件评测函数
│   ├── io_utils.py              # 预测文件写出工具，保持输入空行格式
│   └── filter_zh_chars.py       # fastText 中文单字向量过滤工具
├── results/                     # 模型权重与预测结果
├── requirements.txt             # Python 依赖
└── download_pretrained.sh       # 预训练词向量下载脚本
```

其中 `evaluate_all.py` 是我主要使用的评测入口，可以汇总验证集结果；正式测试集发布后，也可以用同一脚本切换到 test split：

```bash
cd NER
python src/evaluate_all.py val
python src/evaluate_all.py test
```

预测文件统一通过 `io_utils.write_predictions_like_input` 写出。这样做的主要原因是项目说明要求预测文件保持与原输入相同的行数和空行位置，直接按句子循环写文件时容易在末尾多写一个空行，所以我把这部分封装成公共函数，避免不同模型的输出格式不一致。

## 3. 模型设计

### 3.1 HMM

HMM 是我实现的第一个模型，也是整个项目里的生成式基线。我把 NER 看成“隐藏标签序列生成观测 token 序列”的问题：隐藏状态是实体标签，观测是 token。模型主要包含三类概率：

- 初始概率：句首标签分布。
- 转移概率：相邻标签之间的转移分布。
- 发射概率：给定标签生成 token 的概率。

训练时直接根据标注数据统计频数来估计这些概率，并加入加性平滑避免零概率。解码时使用手写 Viterbi 算法，在所有可能标签序列中找得分最高的一条路径。

HMM 的假设比较强：当前标签只依赖前一个标签，当前 token 只依赖当前标签。因此它没有办法像 CRF 一样直接使用“前一个词是否大写”“后一个词是否为机构后缀”这类上下文特征。这个限制会影响效果，但也让模型很容易解释和调试：每一项分数都来自初始概率、转移概率或发射概率，训练过程基本就是统计计数。

在代码实现中，初始概率、转移概率、发射概率都存成 log probability，Viterbi 时只做加法，避免长句概率连乘造成数值下溢。发射概率里还保留了每个标签自己的 `<UNK>` floor，用来处理“这个 token 在训练集中出现过，但没有在当前标签下出现过”的情况。

原始 HMM 对真正 OOV token 的处理比较粗糙，所以我额外加入了 OOV backoff：

- 英文 OOV 使用 `lower / suffix-3 / prefix-3 / word shape` 四类特征的平滑分布插值。
- 中文 OOV 使用粗粒度字符类型（CJK、DIGIT、ALPHA、PUNCT、OTHER）作为 backoff 特征。

这部分仅在 token 从未出现在训练词表时启用；训练集中出现过的 token 仍使用原始发射概率。

这样设计是因为英文实体词经常有比较明显的词形线索，例如人名和地名常见首字母大写，组织机构常有特定后缀，数字和连字符也可能出现在日期、编号或复合实体中。如果只用一个统一的 `<UNK>`，这些信息都会丢失。suffix、prefix 和 shape backoff 不能完全解决 HMM 的上下文缺陷，但能在不改变主模型结构的情况下补充一部分形态信息。中文数据以单字为 token，未登录字符少很多，所以我只使用更粗粒度的字符类型作为兜底。

### 3.2 CRF

CRF 是第二个模型。我使用 `sklearn-crfsuite` 实现线性链条件随机场。和 HMM 不同，CRF 直接建模 `P(y|x)`，也就是在给定句子的情况下预测整条标签序列，因此可以比较自然地加入人工特征。

线性链 CRF 不是逐 token 独立分类，而是在整条标签序列上做全局归一化。它同时利用状态特征和转移特征，所以既能学习“`I-PER` 更应该跟在 `B-PER` 或 `I-PER` 后面”这样的标签依赖，也能利用当前 token 周围的观测信息。

相比 HMM，CRF 不需要为每个标签生成 token 的概率分布，因此不受观测独立假设限制。只要特征对判别有帮助，就可以加入模型。本项目中 CRF 的效果很大程度来自特征工程：英文侧重点是词形和大小写模式，中文侧重点是字符窗口、短语边界和后缀词典。

英文特征包括：

- token lowercase、prefix/suffix 1-4 位；
- 大小写、数字、连字符、标点、字母数字混合等形态特征；
- word shape 与压缩 shape；
- 人名称谓、机构后缀、月份词；
- 前后 1-2 个 token 的上下文特征与组合特征。

中文特征包括：

- 当前字符、字符类型、数字/字母/标点特征；
- `[-3,+3]` 字符窗口；
- bigram / trigram 字符组合；
- 机构后缀、职位后缀、学历词、地名后缀等词典特征。

预测后我还做了轻量的标签修复：英文修复非法 BIO 的 `I-*` 起始问题，中文修复明显非法的 BMES 边界。这一步比较简单，但能避免一些格式层面的错误影响结果。

从实验过程看，CRF 对小数据比较友好。相比随机初始化的神经网络，它的参数与人工特征直接对应，训练过程稳定；只要特征模板覆盖了任务中的关键模式，就能得到较高的验证集表现。中文验证集上 CRF 与 Transformer+CRF、Ensemble 的差距很小，也说明字符级 BMES NER 中，局部字符组合和后缀线索仍然非常强。

### 3.3 Transformer+CRF

第三个模型是 Transformer+CRF。这个模型的目标是让 Transformer 学习上下文表示，再交给 CRF 做全局标签解码。整体结构如下：

1. 可学习 token embedding。
2. 英文 casing embedding，用于表示 lower / upper / title / digit 等大小写模式。
3. char-CNN，用多卷积核抽取 token 内部字符形态特征。
4. 可选静态预训练词向量，与可学习 embedding 拼接后输入模型。
5. sinusoidal positional encoding。
6. PyTorch Transformer Encoder。
7. 手写线性链 CRF 层，负责计算负对数似然和 Viterbi 解码。

CRF 层是手写完成的，包含 forward algorithm 与 Viterbi 解码。我还加入了 BIO/BMES 合法转移约束：英文不允许非法 `I-*` 起始或跨类型延续，中文不允许非法 `M/E` 起始、`B/M` 结束等情况。这样训练和解码时都可以直接屏蔽不合法路径。

预训练向量默认开启：

- 英文使用 GloVe `glove.6B.300d.txt`。
- 中文使用 fastText `cc.zh.300.vec` 过滤出的单字向量 `cc.zh.300.char.vec`。

预训练向量采用 frozen + concat 策略：它不替代可学习 embedding，而是与 128 维可学习 embedding 拼接，作为额外语义先验。

在这个结构中，Transformer Encoder 负责把每个 token 编码为上下文相关表示。与 CRF 的人工窗口特征不同，self-attention 可以直接建模任意距离 token 之间的关系，例如英文实体内部的多词依赖，或中文职位、机构名称中较长的边界信息。位置编码用于补充顺序信息。

我把 CRF 层放在 Transformer 输出之后，是因为 NER 标签并不是彼此独立的。若只使用线性层逐位置分类，模型可能给出 `O I-ORG I-ORG` 或中文 `O M-TITLE E-TITLE` 这类格式非法的序列。CRF 将发射分数和转移分数一起解码，使神经网络的上下文表达与序列标注的结构约束结合起来。训练时，CRF 最大化整条正确标签路径相对所有候选路径的条件概率；预测时，Viterbi 选择全局最优合法路径。

char-CNN 与 casing embedding 是对 token 表示的补充。英文中大小写和词缀高度相关于实体类别，例如全大写缩写常出现在组织机构中，Title Case 常出现在人名和地名中；char-CNN 可以捕捉后缀、前缀和内部字符模式。中文虽然 token 已经是字符，但 char-CNN 仍能为少量非汉字、英文、数字混合 token 提供子字符层面的表示。

预训练向量选择 frozen + concat 而非直接替换可学习 embedding，主要是为了兼顾稳定性和适应性。冻结向量保留 GloVe / fastText 中已有的语义结构，可学习 embedding 则继续适配当前 NER 标签体系。二者拼接后再投影回 `d_model`，相当于让模型同时看到“任务内统计信息”和“外部语义先验”。从后面的消融结果看，这个设计在验证集上带来了稳定收益。

### 3.4 三模型融合

除了三个主模型外，我还在验证集上实现了一个简单 ensemble，用来观察模型之间是否存在互补性：

1. 读取 HMM、CRF、Transformer+CRF 的预测文件。
2. 对每个 token，将各模型预测标签按其验证集 micro F1 加权求和，得到伪 emission 分数。
3. 使用 BIO/BMES 合法转移约束的 Viterbi 重新解码。

该方法不重新训练模型，只利用三个模型不同的偏置进行加权投票，并保证最终输出标签格式合法。

融合方法的直觉是：HMM、CRF、Transformer+CRF 的错误来源不同。HMM 更依赖词表和转移统计，CRF 更依赖人工局部特征，Transformer+CRF 更依赖上下文表示和预训练向量。把三者的输出转化为加权 emission 后再做一次约束 Viterbi，可以在保留强模型意见的同时利用其他模型提供的补充信号。权重采用验证集 F1，是一种简单但可解释的置信度估计。

## 4. 参数设置

### 4.1 HMM 参数

| 参数 | 设置 |
| ---- | ---- |
| 平滑系数 | `1e-6` |
| 英文 OOV backoff | lower 0.45 / suffix 0.30 / prefix 0.15 / shape 0.10 |
| 中文 OOV backoff | chartype 1.00 |
| 解码算法 | Viterbi |

### 4.2 CRF 参数

| 数据集 | `c1` | `c2` | 最大迭代轮数 | 转移设置 |
| ------ | ---- | ---- | ------------ | -------- |
| English | 0.05 | 0.02 | 150 | `all_possible_transitions=True` |
| Chinese | 0.05 | 0.05 | 150 | `all_possible_transitions=True` |

### 4.3 Transformer+CRF 参数

| 参数 | English | Chinese |
| ---- | ------- | ------- |
| `d_model` | 128 | 128 |
| `nhead` | 4 | 4 |
| Transformer layers | 2 | 2 |
| FFN hidden dim | 256 | 256 |
| dropout | 0.45 | 0.35 |
| embedding dropout | 0.20 | 0.10 |
| word dropout | 0.05 | 0.02 |
| char-CNN output dim | 72 | 48 |
| batch size | 48 | 32 |
| learning rate | `8e-4` | `8e-4` |
| weight decay | `1e-4` | `1e-4` |
| max epochs | 100 | 120 |
| early stopping patience | 10 | 18 |
| seed | 42 | 42 |
| pretrained | GloVe-300d | fastText 单字向量 |

优化器使用 AdamW，学习率调度使用 `ReduceLROnPlateau`。因为最终评测指标是 micro F1，所以我没有只按 validation loss 选择模型，而是以验证集 micro F1 作为早停和保存最佳 checkpoint 的主要依据。

## 5. 实验结果

### 5.1 验证集结果

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

验证集上，英文最优为 Ensemble，F1 为 `0.9065`；中文最优为 Transformer+CRF，F1 为 `0.9531`。

### 5.2 预训练向量消融实验

为了确认 GloVe / fastText 静态向量是不是真的有帮助，我把 Transformer+CRF 中的 `use_pretrained` 改为 `False` 后重新训练，其他超参数保持不变。

| 集合 | 模型 | 数据集 | Precision | Recall | F1 | ΔF1 vs 默认 |
| ---- | ---- | ------ | --------- | ------ | --- | ----------- |
| 验证集 | Transformer+CRF | English | 0.9017 | 0.8813 | 0.8914 | -0.0030 |
| 验证集 | Transformer+CRF | Chinese | 0.9522 | 0.9442 | 0.9482 | -0.0049 |

可以看到，预训练向量在验证集上带来了稳定提升，英文提升 `0.0030`，中文提升 `0.0049`。我的理解是，GloVe / fastText 提供了训练集之外的词形和语义先验，尤其能帮助 Transformer+CRF 在 OOV 或低频 token 上少走一些弯路。

## 6. 对比分析

### 6.1 HMM 分析

HMM 是纯生成式基线，优点是实现简单、训练速度快、可解释性强；缺点是独立性假设较强，难以利用复杂上下文特征。加入 OOV backoff 后，英文验证集 F1 从早期基线 `0.7432` 提升到 `0.8369`，提升非常明显。这个结果也符合预期：英文中未登录词比例更高，简单 `<UNK>` 发射概率会丢失大量形态信息，而 suffix、prefix 和 shape 对人名、地名和组织机构都有帮助。

中文 HMM 的提升较小，主要因为中文数据以字符为 token，训练集覆盖率较高，OOV 问题弱于英文；同时中文实体边界更依赖上下文和词组信息，简单字符类型 backoff 的帮助有限。

### 6.2 CRF 分析

CRF 在两个语言上都表现稳定。英文验证集上 CRF 的 F1 达到 `0.9048`，只比 Ensemble 低 `0.0017`；中文验证集上 CRF 的 F1 为 `0.9519`，与 Transformer+CRF 和 Ensemble 都非常接近。这说明人工设计的局部特征和显式转移建模对 NER 任务非常有效，尤其在中文 BMES 标注中，字符窗口、后缀词典和上下文组合特征能够较好捕捉实体边界。

CRF 的主要特点是把模型能力集中在人工定义的局部观察函数上：特征足够贴合数据时，它可以非常高效、稳定；但对于训练集中较少出现、需要语义泛化的新词，CRF 只能依赖词形、上下文窗口和词典提示。因此我在 Transformer+CRF 中额外加入预训练向量，希望弥补 CRF 在语义泛化方面的不足。

### 6.3 Transformer+CRF 分析

Transformer+CRF 的优势在于可以自动学习上下文表示，并通过 CRF 层保证标签序列全局一致。英文验证集上 Transformer+CRF F1 为 `0.8944`，低于 CRF 和 Ensemble，但相比早期版本已经有明显提升；中文验证集上 Transformer+CRF F1 为 `0.9531`，是验证集最优结果。

中文验证集上 Transformer+CRF 以 `0.9531` 首次超过 CRF。这说明当前结构已经能学习到有效的上下文表示，fastText 单字向量和 constrained CRF 对中文 BMES 序列也有帮助。不过 CRF 仍然非常接近，说明中文实体边界中的职位后缀、组织机构后缀等规则性模式，仍然很适合由人工特征显式表达。

从模型分工看，Transformer 层负责产生上下文化 emission，CRF 层负责序列级约束。二者结合后，模型比单纯 Transformer 分类器更适合 NER；消融实验也说明预训练向量并不是简单增加参数，而是在验证集上确实带来了可观察的收益。

### 6.4 Ensemble 分析

Ensemble 在英文验证集上取得最高 F1 `0.9065`，说明三个模型的错误并不完全重合。加权投票倾向于提高 precision：英文 Ensemble precision 为 `0.9195`，高于 CRF 的 `0.9133`，但 recall 略低于 CRF。从这个结果看，融合器整体更保守，会过滤掉一些单模型边缘预测。

中文验证集上 Ensemble F1 为 `0.9530`，与 Transformer+CRF 的 `0.9531` 几乎相同。由于中文 CRF 与 Transformer+CRF 已经非常接近，融合收益有限。

## 7. 实现与调参记录

1. HMM：在基础监督估计和 Viterbi 解码之外，加入 OOV backoff 插值，主要改善英文未登录词预测。
2. CRF：逐步扩展中英文人工特征，并加入 BIO/BMES 修复后处理。
3. Transformer+CRF：实现手写 constrained CRF，加入 char-CNN、casing embedding、F1 early stopping、AdamW、dropout、word dropout。
4. 预训练向量：英文接入 GloVe-300d，中文接入 fastText 单字向量，并通过消融实验确认其贡献。
5. Ensemble：三模型按验证集 F1 加权投票，并使用合法转移约束 Viterbi 重解码。
6. 工程复现：保存模型权重与预测结果，提供 `requirements.txt`、下载脚本、测试脚本和汇总评测脚本。
7. 输出格式：预测文件写出时保持与输入文件相同的空行位置和行数，兼容项目评测脚本要求。

## 8. 总结

本次实验完整实现了 HMM、CRF、Transformer+CRF 三种 NER 模型，并在中英文验证集上进行了训练、评测和对比分析。整体来看，三个模型的特点比较清楚：

- HMM 作为手写生成式基线可解释性强，加入 OOV backoff 后英文表现明显改善。
- CRF 依靠人工特征和转移建模，在两个语言上都非常稳定，英文验证集上只比 Ensemble 略低。
- Transformer+CRF 能自动学习上下文表示，并借助 GloVe / fastText 预训练向量在验证集上获得稳定收益，中文验证集表现最佳。
- 简单加权 Ensemble 能进一步利用模型互补性，在英文验证集上取得最高分。

从验证集结果看，英文可选择 Ensemble，中文可选择 Transformer+CRF。如果正式测试集发布后需要重新选择提交结果，可以直接运行 `src/test_*.py` 生成预测，再用 `src/evaluate_all.py test` 汇总评测。当前报告不包含正式测试集分数。
