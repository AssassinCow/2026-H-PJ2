"""
Task 2: CRF-based Named Entity Recognition
Uses sklearn-crfsuite for sequence labeling on both Chinese and English NER datasets.
"""

import os
import re
import sys
import unicodedata
import warnings
warnings.filterwarnings("ignore")

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics


# ============================================================
# Data loading
# ============================================================

def load_data(filepath):
    """Load NER data file. Returns list of sentences, each sentence is a list of (token, tag) tuples."""
    sentences = []
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split(' ')
                if len(parts) == 2:
                    token, tag = parts
                    sentence.append((token, tag))
                elif len(parts) == 1:
                    # prediction mode: only token, no tag
                    sentence.append((parts[0], 'O'))
        if sentence:
            sentences.append(sentence)
    return sentences


# ============================================================
# Feature extraction
# ============================================================

PERSON_TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'lord', 'lady', 'president',
}
ORG_SUFFIXES = {
    'inc', 'corp', 'co', 'ltd', 'llc', 'plc', 'group', 'bank', 'university',
    'college', 'committee', 'agency', 'ministry', 'department', 'team',
}
MONTHS = {
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
}

CN_ORG_SUFFIXES = (
    '公司', '集团', '大学', '学院', '医院', '银行', '中心', '研究所', '委员会',
    '办公室', '学校', '协会', '政府', '部门', '厂', '局', '部', '厅', '所',
)
CN_TITLE_SUFFIXES = (
    '经理', '主管', '主任', '工程师', '教授', '老师', '顾问', '专员', '总监',
    '助理', '博士', '硕士', '研究员', '会计', '律师', '医师',
)
CN_EDU_WORDS = ('博士', '硕士', '本科', '大专', '高中', '学士', '研究生')
CN_LOC_SUFFIXES = ('省', '市', '区', '县', '镇', '乡', '村', '路', '街')


def word_shape(word):
    """Map surface form to a compact shape, e.g. McDonalds -> XxXxxxxxx."""
    chars = []
    for ch in word:
        if ch.isupper():
            chars.append('X')
        elif ch.islower():
            chars.append('x')
        elif ch.isdigit():
            chars.append('d')
        elif ch in '-_':
            chars.append('-')
        elif ch in '.,':
            chars.append('.')
        else:
            chars.append(ch)
    return ''.join(chars)


def compressed_shape(word):
    return re.sub(r'(.)\1+', r'\1', word_shape(word))


def char_type(ch):
    category = unicodedata.category(ch)
    if ch.isdigit():
        return 'digit'
    if '\u4e00' <= ch <= '\u9fff':
        return 'cjk'
    if ch.isalpha():
        return 'alpha'
    if category.startswith('P'):
        return 'punct'
    if category.startswith('S'):
        return 'symbol'
    return category[:1]


def add_cn_suffix_features(features, sent, i):
    text = ''.join(token for token, _ in sent)
    for name, suffixes in (
        ('org_suffix', CN_ORG_SUFFIXES),
        ('title_suffix', CN_TITLE_SUFFIXES),
        ('edu_word', CN_EDU_WORDS),
        ('loc_suffix', CN_LOC_SUFFIXES),
    ):
        for suffix in suffixes:
            start = i - len(suffix) + 1
            if start >= 0 and text[start:i + 1] == suffix:
                features[f'ends_{name}={suffix}'] = True
            if text[i:i + len(suffix)] == suffix:
                features[f'begins_{name}={suffix}'] = True


def word_features_en(sent, i):
    """Extract features for English word at position i in sentence."""
    word = sent[i][0]
    lower = word.lower()
    features = {
        'bias': 1.0,
        'word.lower()': lower,
        'word.shape': word_shape(word),
        'word.shape.compressed': compressed_shape(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'word.isalnum()': word.isalnum(),
        'word.has_hyphen': '-' in word,
        'word.has_digit': any(ch.isdigit() for ch in word),
        'word.has_alpha': any(ch.isalpha() for ch in word),
        'word.has_period': '.' in word,
        'word.has_slash': '/' in word,
        'word.has_apostrophe': "'" in word,
        'word.is_person_title': lower.rstrip('.') in PERSON_TITLES,
        'word.is_org_suffix': lower.rstrip('.') in ORG_SUFFIXES,
        'word.is_month': lower.rstrip('.') in MONTHS,
        'word.length': len(word),
    }
    for n in range(1, 5):
        features[f'prefix{n}'] = lower[:n]
        features[f'suffix{n}'] = lower[-n:]

    if i > 0:
        word1 = sent[i - 1][0]
        lower1 = word1.lower()
        features.update({
            '-1:word.lower()': lower1,
            '-1:word.shape': compressed_shape(word1),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.is_person_title': lower1.rstrip('.') in PERSON_TITLES,
            '-1:word.is_org_suffix': lower1.rstrip('.') in ORG_SUFFIXES,
            '-1/current.lower': lower1 + '/' + lower,
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i > 1:
        word2 = sent[i - 2][0]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.shape': compressed_shape(word2),
            '-2:word.istitle()': word2.istitle(),
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        lower1 = word1.lower()
        features.update({
            '+1:word.lower()': lower1,
            '+1:word.shape': compressed_shape(word1),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.is_org_suffix': lower1.rstrip('.') in ORG_SUFFIXES,
            'current/+1.lower': lower + '/' + lower1,
        })
    else:
        features['EOS'] = True  # End of sentence

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.shape': compressed_shape(word2),
            '+2:word.istitle()': word2.istitle(),
        })

    # Bigram features
    if i > 0:
        features['-1:word/word'] = sent[i - 1][0].lower() + '/' + lower
    if i < len(sent) - 1:
        features['word/+1:word'] = lower + '/' + sent[i + 1][0].lower()

    return features


def word_features_cn(sent, i):
    """Extract features for Chinese character at position i in sentence."""
    char = sent[i][0]
    features = {
        'bias': 1.0,
        'char': char,
        'char.type': char_type(char),
        'char.is_digit': char.isdigit(),
        'char.is_alpha': char.isalpha(),
    }

    # Punctuation check
    features['char.is_punct'] = unicodedata.category(char).startswith('P')

    # Context window [-2, -1, 0, +1, +2] — matches the template_for_crf.utf8
    for delta in [-3, -2, -1, 1, 2, 3]:
        pos = i + delta
        if 0 <= pos < len(sent):
            c = sent[pos][0]
            features[f'{delta}:char'] = c
            features[f'{delta}:char.type'] = char_type(c)
        else:
            if delta < 0:
                features[f'{delta}:BOS'] = True
            else:
                features[f'{delta}:EOS'] = True

    # Bigram features (matching template)
    if i >= 2:
        features['-2/-1:char'] = sent[i - 2][0] + '/' + sent[i - 1][0]
    if i >= 1:
        features['-1/0:char'] = sent[i - 1][0] + '/' + char
    if i >= 1 and i < len(sent) - 1:
        features['-1/+1:char'] = sent[i - 1][0] + '/' + sent[i + 1][0]
    if i < len(sent) - 1:
        features['0/+1:char'] = char + '/' + sent[i + 1][0]
    if i < len(sent) - 2:
        features['+1/+2:char'] = sent[i + 1][0] + '/' + sent[i + 2][0]
    if i >= 1 and i < len(sent) - 1:
        features['-1/0/+1:char'] = sent[i - 1][0] + '/' + char + '/' + sent[i + 1][0]

    add_cn_suffix_features(features, sent, i)

    # BOS/EOS markers
    if i == 0:
        features['BOS'] = True
    if i == len(sent) - 1:
        features['EOS'] = True

    return features


def sent2features(sent, language):
    feat_fn = word_features_cn if language == 'Chinese' else word_features_en
    return [feat_fn(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [tag for _, tag in sent]


def sent2tokens(sent):
    return [token for token, _ in sent]


def repair_bio_sequence(tags):
    repaired = []
    prev_type = None
    prev_prefix = 'O'
    for tag in tags:
        if tag == 'O' or '-' not in tag:
            repaired.append(tag)
            prev_prefix, prev_type = 'O', None
            continue
        prefix, ent_type = tag.split('-', 1)
        if prefix == 'I' and (prev_prefix == 'O' or prev_type != ent_type):
            tag = f'B-{ent_type}'
            prefix = 'B'
        repaired.append(tag)
        prev_prefix, prev_type = prefix, ent_type
    return repaired


def repair_bmes_sequence(tags):
    repaired = list(tags)
    n = len(repaired)
    for i, tag in enumerate(repaired):
        if tag == 'O' or '-' not in tag:
            continue
        prefix, ent_type = tag.split('-', 1)
        prev_tag = repaired[i - 1] if i > 0 else 'O'
        next_tag = repaired[i + 1] if i + 1 < n else 'O'
        prev_ok = prev_tag in {f'B-{ent_type}', f'M-{ent_type}'}
        next_ok = next_tag in {f'M-{ent_type}', f'E-{ent_type}'}

        if prefix == 'B' and not next_ok:
            repaired[i] = f'S-{ent_type}'
        elif prefix == 'M':
            if not prev_ok and next_ok:
                repaired[i] = f'B-{ent_type}'
            elif prev_ok and not next_ok:
                repaired[i] = f'E-{ent_type}'
            elif not prev_ok and not next_ok:
                repaired[i] = f'S-{ent_type}'
        elif prefix == 'E' and not prev_ok:
            repaired[i] = f'S-{ent_type}'
    return repaired


def repair_sequence(tags, language):
    if language == 'English':
        return repair_bio_sequence(tags)
    return repair_bmes_sequence(tags)


CRF_CONFIG = {
    'English': dict(c1=0.05, c2=0.02, max_iterations=150),
    'Chinese': dict(c1=0.05, c2=0.05, max_iterations=150),
}


# ============================================================
# Training and prediction
# ============================================================

def train_and_predict(language, data_dir, output_path):
    """Train CRF on training set and predict on validation set."""
    train_path = os.path.join(data_dir, language, 'train.txt')
    val_path = os.path.join(data_dir, language, 'validation.txt')

    print(f"[{language}] Loading data...")
    train_sents = load_data(train_path)
    val_sents = load_data(val_path)
    print(f"[{language}] Train: {len(train_sents)} sentences, Val: {len(val_sents)} sentences")

    print(f"[{language}] Extracting features...")
    X_train = [sent2features(s, language) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_val = [sent2features(s, language) for s in val_sents]
    y_val = [sent2labels(s) for s in val_sents]

    print(f"[{language}] Training CRF model...")
    cfg = CRF_CONFIG.get(language, CRF_CONFIG['English'])
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=cfg['c1'],
        c2=cfg['c2'],
        max_iterations=cfg['max_iterations'],
        all_possible_transitions=True,
        verbose=False,
    )
    crf.fit(X_train, y_train)

    print(f"[{language}] Predicting...")
    y_pred = crf.predict(X_val)
    y_pred = [repair_sequence(seq, language) for seq in y_pred]

    # Write prediction file in the required format
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(val_sents):
            tokens = sent2tokens(sent)
            preds = y_pred[sent_idx]
            for token, pred_tag in zip(tokens, preds):
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")

    print(f"[{language}] Predictions written to {output_path}")

    # Evaluate
    labels = sorted({
        tag
        for seq in y_val + y_pred
        for tag in seq
        if tag != 'O'
    })

    print()
    print("=" * 70)
    print(f"  [{language}] CRF — Per-tag Classification Report")
    print("=" * 70)
    print(crf_metrics.flat_classification_report(y_val, y_pred, labels=labels, digits=4))

    # Compute micro avg F1 manually for the highlighted summary box
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for gold_seq, pred_seq in zip(y_val, y_pred):
        for g, p in zip(gold_seq, pred_seq):
            if g == 'O' and p == 'O':
                continue
            if g == p:
                tp[g] += 1
            else:
                if p != 'O':
                    fp[p] += 1
                if g != 'O':
                    fn[g] += 1
    total_tp = sum(tp.values()); total_fp = sum(fp.values()); total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print("+" + "=" * 62 + "+")
    title = f">>> [{language}] CRF Final Score (micro avg)"
    print(f"|  {title}" + " " * (62 - 2 - len(title)) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Precision: {micro_p:.4f}   Recall: {micro_r:.4f}   F1: {micro_f1:.4f}   " + " " * 6 + "|")
    print("+" + "=" * 62 + "+")
    print()

    return crf, (micro_p, micro_r, micro_f1)


def predict_test(crf, language, test_path, output_path):
    """Predict on a test file (for interview use)."""
    test_sents = load_data(test_path)
    X_test = [sent2features(s, language) for s in test_sents]
    y_pred = crf.predict(X_test)
    y_pred = [repair_sequence(seq, language) for seq in y_pred]

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(test_sents):
            tokens = sent2tokens(sent)
            preds = y_pred[sent_idx]
            for token, pred_tag in zip(tokens, preds):
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"[{language}] Test predictions written to {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Default: run both languages
    languages = ['English', 'Chinese']
    if len(sys.argv) > 1:
        languages = [sys.argv[1]]

    summary = {}
    for lang in languages:
        output_file = os.path.join(data_dir, f'crf_result_{lang.lower()}.txt')
        model, metrics = train_and_predict(lang, data_dir, output_file)
        summary[lang] = metrics

        # If test.txt exists, also predict on it
        test_path = os.path.join(data_dir, lang, 'test.txt')
        if os.path.exists(test_path):
            test_output = os.path.join(data_dir, f'crf_test_result_{lang.lower()}.txt')
            predict_test(model, lang, test_path, test_output)

    if len(summary) > 1:
        print()
        print("#" * 64)
        print("#  CRF — Overall Summary (micro avg on validation)")
        print("#" * 64)
        print(f"  {'Language':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
        print("  " + "-" * 50)
        for lang, (p, r, f1) in summary.items():
            print(f"  {lang:<12s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}")
        print("#" * 64)
