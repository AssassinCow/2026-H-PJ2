# 2026-H-PJ2 · 命名实体识别（NER）

复旦大学《人工智能》课程 Project 2（H 类）。学号 24300240170 / 刘卓鑫。

任务：在英文（CoNLL-2003 风格）与中文（简历语料）两套数据集上，分别实现并对比三种序列标注模型用于命名实体识别。

## 目录结构

```
2026-H-PJ2/
├── README.md                       # 本文件：项目总览
├── pj2H.pdf                        # 课程作业题目说明
├── NER.rar                         # 原始数据/代码骨架压缩包（本地保留，已 gitignore）
└── NER/                            # 主代码目录（详见 NER/README.md）
    ├── README.md                   # 环境配置 / 各任务运行命令 / 评测结果
    ├── requirements.txt            # Python 依赖列表
    ├── download_pretrained.sh      # GloVe / fastText 预训练词向量下载脚本
    ├── template_for_crf.utf8       # CRF++ 风格特征模板（参考用）
    ├── data/                       # 数据集
    │   ├── English/                # 英文：train.txt / validation.txt / test.txt / tag.txt
    │   ├── Chinese/                # 中文：train.txt / validation.txt / test.txt / tag.txt
    │   └── example_data/           # 评测格式示例
    ├── pretrained/                 # GloVe / fastText 词向量（gitignore，下载脚本生成）
    ├── src/                        # 源代码
    │   ├── hmm_ner.py              # 任务 1：手写 HMM（Viterbi 解码）
    │   ├── crf_ner.py              # 任务 2：CRF（sklearn-crfsuite）
    │   ├── transformer_crf_ner.py  # 任务 3：Transformer + 手写 CRF（PyTorch）
    │   ├── ensemble.py             # 三模型加权投票融合 + BIO/BMES 约束 Viterbi
    │   ├── test_*.py               # 各模型加载 checkpoint 跑 test.txt 的脚本
    │   ├── check.py                # 单文件评测脚本（sklearn classification_report）
    │   ├── evaluate_all.py         # 一键扫描所有结果文件并汇总对比（支持 val/test）
    │   └── filter_zh_chars.py      # fastText 中文字向量过滤工具
    └── results/                    # 各模型权重 + 验证集预测（test 预测已 gitignore）
        ├── hmm/                    # hmm_model_{lang}.pkl + hmm_result_{lang}.txt
        ├── crf/                    # crf_model_{lang}.pkl  + crf_result_{lang}.txt
        ├── transformer_crf/        # transformer_crf_checkpoint_{lang}.pt + 预测
        └── ensemble/               # ensemble_result_{lang}.txt
```

## 三个任务对照

| 任务 | 模型 | 实现方式 | 主入口 |
| ---- | ---- | -------- | ------ |
| 1 | HMM | 纯 Python 手写，监督估计参数 + Viterbi 解码 | [NER/src/hmm_ner.py](NER/src/hmm_ner.py) |
| 2 | CRF | sklearn-crfsuite，配合人工设计的字/词级特征 | [NER/src/crf_ner.py](NER/src/crf_ner.py) |
| 3 | Transformer+CRF | PyTorch Transformer Encoder + 约束 CRF；英文加入 casing embedding 与 char-CNN 并拼接 GloVe-300d 冻结向量；中文加入 char-CNN 与 fastText `cc.zh.300` 单字向量 | [NER/src/transformer_crf_ner.py](NER/src/transformer_crf_ner.py) |
| 融合 | Ensemble | 三模型按验证集 F1 加权投票得到伪 emission，再用合法 BIO/BMES 转移约束 Viterbi 重解码 | [NER/src/ensemble.py](NER/src/ensemble.py) |

## 当前结论

验证集 token-level micro F1：

- 英文最高分来自三模型融合 Ensemble `0.9065`，单模型最优 CRF `0.9048` 紧随其后；增强后的 Transformer+CRF（接入 GloVe-300d 后）`0.8944`。
- 中文最高分首次由单模型 Transformer+CRF 取得 `0.9531`（拼接 fastText 单字向量），与 Ensemble `0.9530` 几乎贴着，CRF `0.9519`。

提交策略：英文优先 Ensemble；中文优先 Transformer+CRF（也可直接选 Ensemble，差距约 `0.0001`）。若要求单模型一致提交，CRF 在两个语言上都是稳健次选。完整验证集实验表、ablation 和优化说明见 [NER/README.md](NER/README.md)。

## 数据格式

每个数据文件按行存储 `token tag`，句子之间用空行分隔。

- 英文标签集（9 类）：`O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC / B-MISC / I-MISC`
- 中文标签集（33 类，BMES 模式）：`O` 加 8 类实体（NAME/CONT/EDU/TITLE/ORG/RACE/PRO/LOC）的 `B-/M-/E-/S-` 标记

详见 [NER/data/English/tag.txt](NER/data/English/tag.txt)、[NER/data/Chinese/tag.txt](NER/data/Chinese/tag.txt)。

## 快速开始

环境配置、运行命令、评测方式与当前实验结果都在子目录 README 中：

➡  [NER/README.md](NER/README.md)

## 测试集复现

正式 `test.txt` 公布后，放入 `NER/data/English/` 或 `NER/data/Chinese/` 目录，运行各 `src/test_*.py` 加载已保存的权重直接预测（**无需重训**），结果写入 `NER/results/<model>/<model>_test_result_<lang>.txt`，再用 `python src/evaluate_all.py test` 统一汇总评测。详见 [NER/README.md](NER/README.md) 的"测试"与"评测"小节。
