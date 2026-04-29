# 2026-H-Pj2 · 命名实体识别（NER）

复旦大学《人工智能》课程 Project 2（H 类）。学号 24300240170 / 刘卓欣。

任务：在英文（CoNLL-2003 风格）与中文（简历语料）两套数据集上，分别实现并对比三种序列标注模型用于命名实体识别。

## 目录结构

```
2026-H-Pj2/
├── README.md               # 本文件：项目总览
├── pj2H.pdf                # 课程作业题目说明
├── NER.rar                 # 原始数据/代码骨架压缩包（来自课程发布）
└── NER/                    # 主代码目录（详见 NER/README.md）
    ├── README.md           # 环境配置 / 各任务运行命令 / 评测结果
    ├── hmm_ner.py          # 任务 1：手写 HMM（Viterbi 解码）
    ├── crf_ner.py          # 任务 2：CRF（sklearn-crfsuite）
    ├── transformer_crf_ner.py  # 任务 3：Transformer + 手写 CRF（PyTorch）
    ├── check.py            # 单文件评测脚本（sklearn classification_report）
    ├── evaluate_all.py     # 一键扫描所有结果文件并汇总对比
    ├── template_for_crf.utf8   # CRF++ 风格特征模板（参考用）
    ├── English/            # 英文数据：train.txt / validation.txt / tag.txt
    ├── Chinese/            # 中文数据：train.txt / validation.txt / tag.txt
    ├── example_data/       # 评测格式示例
    └── *_result_*.txt      # 各模型在验证集上的预测输出
```

## 三个任务对照

| 任务 | 模型 | 实现方式 | 主入口 |
| ---- | ---- | -------- | ------ |
| 1 | HMM | 纯 Python 手写，监督估计参数 + Viterbi 解码 | [NER/hmm_ner.py](NER/hmm_ner.py) |
| 2 | CRF | sklearn-crfsuite，配合人工设计的字/词级特征 | [NER/crf_ner.py](NER/crf_ner.py) |
| 3 | Transformer+CRF | PyTorch 实现的 Transformer Encoder + 手写 CRF 层；英文额外加入 casing embedding 与 char-CNN | [NER/transformer_crf_ner.py](NER/transformer_crf_ner.py) |

## 数据格式

每个数据文件按行存储 `token tag`，句子之间用空行分隔。

- 英文标签集（9 类）：`O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC / B-MISC / I-MISC`
- 中文标签集（33 类，BMES 模式）：`O` 加 8 类实体（NAME/CONT/EDU/TITLE/ORG/RACE/PRO/LOC）的 `B-/M-/E-/S-` 标记

详见 [NER/English/tag.txt](NER/English/tag.txt)、[NER/Chinese/tag.txt](NER/Chinese/tag.txt)。

## 快速开始

环境配置、运行命令、评测方式与当前实验结果都在子目录 README 中：

➡  [NER/README.md](NER/README.md)

## 面试 / 测试集复现

提交方拿到 `test.txt` 后，将其放入 `NER/English/` 或 `NER/Chinese/` 目录，重新运行对应脚本即可生成 `*_result_*.txt` 预测文件，再用 `check.py` 或 `evaluate_all.py` 评测。
