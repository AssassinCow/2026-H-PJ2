"""
Task 3: Transformer+CRF Named Entity Recognition (optimized)

Architecture (per language):
- Word embedding (always)
- Casing feature embedding (English only — captures Title/UPPER/lower/digit/punct)
- Character-level CNN (English only — captures suffixes/prefixes like CRF's word[-3:])
- Sinusoidal positional encoding
- Transformer encoder (PyTorch)
- Hand-written CRF layer (forward algo + Viterbi decoding)

Key training tricks:
- Early stopping with patience to avoid wasted compute and overfit
- Different dropout / lr-scheduling per language
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict


# ============================================================
# Casing features (Lample et al. 2016 style)
# ============================================================

CASING_PAD = 0
CASING_LOWER = 1
CASING_UPPER = 2
CASING_TITLE = 3
CASING_DIGIT = 4
CASING_MIXED_NUM = 5   # contains digits + letters
CASING_PUNCT = 6
CASING_OTHER = 7
NUM_CASINGS = 8


def get_casing(word):
    if word.isdigit():
        return CASING_DIGIT
    if word.isalpha():
        if word.islower():
            return CASING_LOWER
        if word.isupper():
            return CASING_UPPER
        if word.istitle():
            return CASING_TITLE
        return CASING_OTHER
    has_digit = any(c.isdigit() for c in word)
    has_alpha = any(c.isalpha() for c in word)
    if has_digit and has_alpha:
        return CASING_MIXED_NUM
    if not any(c.isalnum() for c in word):
        return CASING_PUNCT
    return CASING_OTHER


# ============================================================
# Data loading and preprocessing
# ============================================================

def load_data(filepath):
    """Load NER data. Returns list of sentences, each a list of (token, tag)."""
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
                    sentence.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    sentence.append((parts[0], 'O'))
        if sentence:
            sentences.append(sentence)
    return sentences


def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sent in sentences:
        for token, _ in sent:
            counter[token] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def build_char_vocab(sentences):
    """Char-level vocab from training sentences (only used for English char-CNN)."""
    chars = set()
    for sent in sentences:
        for token, _ in sent:
            for ch in token:
                chars.add(ch)
    char2idx = {'<PAD>': 0, '<UNK>': 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)
    return char2idx


def build_tag_map(sentences):
    tags = set()
    for sent in sentences:
        for _, tag in sent:
            tags.add(tag)
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag


class NERDataset(Dataset):
    def __init__(self, sentences, vocab, tag2idx, char2idx=None, use_casing=False,
                 max_word_len=20):
        self.sentences = sentences
        self.vocab = vocab
        self.tag2idx = tag2idx
        self.char2idx = char2idx
        self.use_casing = use_casing
        self.use_char = char2idx is not None
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        token_ids = [self.vocab.get(t, self.vocab['<UNK>']) for t, _ in sent]
        tag_ids = [self.tag2idx[tag] for _, tag in sent]

        casing_ids = None
        if self.use_casing:
            casing_ids = [get_casing(t) for t, _ in sent]

        char_ids = None
        if self.use_char:
            unk = self.char2idx['<UNK>']
            char_ids = []
            for t, _ in sent:
                cs = [self.char2idx.get(ch, unk) for ch in t[:self.max_word_len]]
                char_ids.append(cs)

        return {
            'tokens': torch.tensor(token_ids, dtype=torch.long),
            'tags': torch.tensor(tag_ids, dtype=torch.long),
            'casings': torch.tensor(casing_ids, dtype=torch.long) if casing_ids is not None else None,
            'chars': char_ids,  # list of lists, padded later
            'length': len(sent),
        }


def make_collate_fn(use_casing=False, use_char=False, max_word_len=20):
    def collate(batch):
        lengths = [b['length'] for b in batch]
        max_len = max(lengths)
        bs = len(batch)

        padded_tokens = torch.zeros(bs, max_len, dtype=torch.long)
        padded_tags = torch.zeros(bs, max_len, dtype=torch.long)
        mask = torch.zeros(bs, max_len, dtype=torch.bool)

        padded_casings = torch.zeros(bs, max_len, dtype=torch.long) if use_casing else None
        padded_chars = torch.zeros(bs, max_len, max_word_len, dtype=torch.long) if use_char else None

        for i, b in enumerate(batch):
            l = b['length']
            padded_tokens[i, :l] = b['tokens']
            padded_tags[i, :l] = b['tags']
            mask[i, :l] = True
            if use_casing:
                padded_casings[i, :l] = b['casings']
            if use_char:
                for j, cs in enumerate(b['chars']):
                    padded_chars[i, j, :len(cs)] = torch.tensor(cs, dtype=torch.long)

        return padded_tokens, padded_tags, mask, padded_casings, padded_chars

    return collate


# ============================================================
# Hand-written CRF layer
# ============================================================

class CRF(nn.Module):
    """Linear-chain CRF: forward algorithm + Viterbi decoding (hand-written)."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_len):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score += (trans + emit) * mask[:, t].float()
        last_positions = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            scores = alpha.unsqueeze(2) + trans + emit
            new_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, t].unsqueeze(1).float()
            alpha = new_alpha * m + alpha * (1 - m)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        log_Z = self._compute_log_partition(emissions, mask)
        score = self._compute_score(emissions, tags, mask)
        return (log_Z - score).mean()

    def viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for t in range(1, seq_len):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=1)
            new_viterbi = best_scores + emissions[:, t]
            m = mask[:, t].unsqueeze(1).float()
            viterbi = new_viterbi * m + viterbi * (1 - m)
            backpointers.append(best_tags)
        viterbi += self.end_transitions.unsqueeze(0)
        lengths = mask.long().sum(dim=1)
        _, best_last_tags = viterbi.max(dim=1)
        best_paths = []
        for b in range(batch_size):
            seq_len_b = lengths[b].item()
            path = [0] * seq_len_b
            path[-1] = best_last_tags[b].item()
            for t in range(seq_len_b - 2, -1, -1):
                path[t] = backpointers[t][b][path[t + 1]].item()
            best_paths.append(path)
        return best_paths


# ============================================================
# Model
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CharCNN(nn.Module):
    """Character-level CNN: per-word character representation via 1D conv + max-pool."""
    def __init__(self, num_chars, d_char_emb=25, d_char_out=30, kernel_size=3):
        super().__init__()
        self.char_emb = nn.Embedding(num_chars, d_char_emb, padding_idx=0)
        self.conv = nn.Conv1d(d_char_emb, d_char_out, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, chars):
        # chars: (batch, seq_len, max_word_len)
        b, s, w = chars.shape
        x = self.char_emb(chars.view(-1, w))   # (b*s, w, d_char_emb)
        x = self.dropout(x)
        x = x.transpose(1, 2)                  # (b*s, d_char_emb, w)
        x = F.relu(self.conv(x))               # (b*s, d_char_out, w)
        x = x.max(dim=2)[0]                    # (b*s, d_char_out)
        return x.view(b, s, -1)                # (b, s, d_char_out)


class TransformerCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.3,
                 # Optional features
                 use_casing=False, num_casings=NUM_CASINGS, d_case=16,
                 use_char_cnn=False, num_chars=0, d_char_emb=25, d_char_out=30):
        super().__init__()
        self.use_casing = use_casing
        self.use_char_cnn = use_char_cnn

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Optional feature components
        feat_dim = d_model
        if use_casing:
            self.case_emb = nn.Embedding(num_casings, d_case, padding_idx=0)
            feat_dim += d_case
        if use_char_cnn:
            self.char_cnn = CharCNN(num_chars, d_char_emb, d_char_out)
            feat_dim += d_char_out

        # Project concatenated features back to d_model when needed
        if feat_dim != d_model:
            self.input_proj = nn.Linear(feat_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden2tag = nn.Linear(d_model, num_tags)
        self.crf = CRF(num_tags)
        self.dropout = nn.Dropout(dropout)

    def _get_emissions(self, tokens, mask, casings=None, chars=None):
        x = self.embedding(tokens)
        parts = [x]
        if self.use_casing:
            parts.append(self.case_emb(casings))
        if self.use_char_cnn:
            parts.append(self.char_cnn(chars))
        if len(parts) > 1:
            x = torch.cat(parts, dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.dropout(x)
        return self.hidden2tag(x)

    def loss(self, tokens, tags, mask, casings=None, chars=None):
        emissions = self._get_emissions(tokens, mask, casings, chars)
        return self.crf.neg_log_likelihood(emissions, tags, mask)

    def predict(self, tokens, mask, casings=None, chars=None):
        emissions = self._get_emissions(tokens, mask, casings, chars)
        return self.crf.viterbi_decode(emissions, mask)


# ============================================================
# Training utilities
# ============================================================

# Per-language hyperparameter config
LANG_CONFIG = {
    'English': dict(
        use_casing=True,
        use_char_cnn=True,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.5,           # higher dropout — English overfits hard
        epochs=80,
        batch_size=64,
        lr=1e-3,
        patience=8,            # early-stop after 8 epochs without improvement
    ),
    'Chinese': dict(
        use_casing=False,
        use_char_cnn=False,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3,
        epochs=120,
        batch_size=64,
        lr=1e-3,
        patience=15,
    ),
}


def run_validation(model, val_loader, device):
    model.eval()
    val_loss = 0
    n = 0
    with torch.no_grad():
        for tokens, tags, mask, casings, chars in val_loader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            casings = casings.to(device) if casings is not None else None
            chars = chars.to(device) if chars is not None else None
            loss = model.loss(tokens, tags, mask, casings, chars)
            val_loss += loss.item()
            n += 1
    return val_loss / max(n, 1)


def train_and_predict(language, data_dir, output_path, device='cpu'):
    cfg = LANG_CONFIG[language]
    train_path = os.path.join(data_dir, language, 'train.txt')
    val_path = os.path.join(data_dir, language, 'validation.txt')

    print(f"[{language}] Loading data...")
    train_sents = load_data(train_path)
    val_sents = load_data(val_path)
    print(f"[{language}] Train: {len(train_sents)} sentences, Val: {len(val_sents)} sentences")

    # Vocab + tags
    vocab = build_vocab(train_sents, min_freq=1)
    tag2idx, idx2tag = build_tag_map(train_sents)
    char2idx = build_char_vocab(train_sents) if cfg['use_char_cnn'] else None
    print(f"[{language}] Vocab: {len(vocab)}, Tags: {len(tag2idx)}, "
          f"Chars: {len(char2idx) if char2idx else 'N/A'}, "
          f"Casing: {cfg['use_casing']}, CharCNN: {cfg['use_char_cnn']}")

    # Datasets / loaders
    use_casing = cfg['use_casing']
    use_char = cfg['use_char_cnn']
    train_ds = NERDataset(train_sents, vocab, tag2idx, char2idx, use_casing)
    val_ds = NERDataset(val_sents, vocab, tag2idx, char2idx, use_casing)
    collate = make_collate_fn(use_casing=use_casing, use_char=use_char)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate)

    # Model
    model = TransformerCRF(
        vocab_size=len(vocab),
        num_tags=len(tag2idx),
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        dim_feedforward=cfg['dim_feedforward'],
        dropout=cfg['dropout'],
        use_casing=use_casing,
        use_char_cnn=use_char,
        num_chars=len(char2idx) if char2idx else 0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{language}] Model params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"[{language}] Training on {device} (epochs<={cfg['epochs']}, patience={cfg['patience']})...")
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for tokens, tags, mask, casings, chars in train_loader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            casings = casings.to(device) if casings is not None else None
            chars = chars.to(device) if chars is not None else None
            optimizer.zero_grad()
            loss = model.loss(tokens, tags, mask, casings, chars)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        val_loss = run_validation(model, val_loader, device)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
            marker = "  *"
        else:
            no_improve += 1

        print(f"  Epoch {epoch:3d}/{cfg['epochs']}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={cur_lr:.6f}{marker}")

        if no_improve >= cfg['patience']:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, val_loss {best_val_loss:.4f})")
            break

    # Restore best
    model.load_state_dict(best_state)
    model.to(device)
    print(f"[{language}] Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # Predict on validation
    print(f"[{language}] Predicting...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for tokens, tags, mask, casings, chars in val_loader:
            tokens, mask = tokens.to(device), mask.to(device)
            casings = casings.to(device) if casings is not None else None
            chars = chars.to(device) if chars is not None else None
            paths = model.predict(tokens, mask, casings, chars)
            all_preds.extend(paths)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(val_sents):
            preds = all_preds[sent_idx]
            for i, (token, _) in enumerate(sent):
                pred_tag = idx2tag[preds[i]]
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"[{language}] Predictions written to {output_path}")

    metrics = evaluate(language, val_sents, all_preds, idx2tag)
    return model, vocab, tag2idx, idx2tag, char2idx, metrics


def evaluate(language, val_sents, all_preds, idx2tag):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for sent_idx, sent in enumerate(val_sents):
        preds = all_preds[sent_idx]
        for i, (_, gold_tag) in enumerate(sent):
            pred_tag = idx2tag[preds[i]]
            if gold_tag == 'O' and pred_tag == 'O':
                continue
            if gold_tag == pred_tag:
                tp[gold_tag] += 1
            else:
                if pred_tag != 'O':
                    fp[pred_tag] += 1
                if gold_tag != 'O':
                    fn[gold_tag] += 1

    all_tags = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    total_tp = sum(tp[t] for t in all_tags)
    total_fp = sum(fp[t] for t in all_tags)
    total_fn = sum(fn[t] for t in all_tags)

    print()
    print("=" * 64)
    print(f"  [{language}] Transformer+CRF — Per-tag results")
    print("=" * 64)
    print(f"  {'Tag':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>8s}")
    print("  " + "-" * 60)
    for tag in all_tags:
        p = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0
        r = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[tag] + fn[tag]
        print(f"  {tag:<12s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}  {support:8d}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print()
    print("+" + "=" * 62 + "+")
    title = f">>> [{language}] Transformer+CRF Final Score (micro avg)"
    print(f"|  {title}" + " " * max(0, 62 - 2 - len(title)) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Precision: {micro_p:.4f}   Recall: {micro_r:.4f}   F1: {micro_f1:.4f}   " + " " * 6 + "|")
    print("+" + "=" * 62 + "+")
    print()
    return micro_p, micro_r, micro_f1


def predict_test(model, vocab, tag2idx, idx2tag, char2idx, language, test_path,
                 output_path, device='cpu'):
    cfg = LANG_CONFIG[language]
    test_sents = load_data(test_path)
    test_ds = NERDataset(test_sents, vocab, tag2idx, char2idx, cfg['use_casing'])
    collate = make_collate_fn(use_casing=cfg['use_casing'], use_char=cfg['use_char_cnn'])
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for tokens, tags, mask, casings, chars in test_loader:
            tokens, mask = tokens.to(device), mask.to(device)
            casings = casings.to(device) if casings is not None else None
            chars = chars.to(device) if chars is not None else None
            paths = model.predict(tokens, mask, casings, chars)
            all_preds.extend(paths)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(test_sents):
            preds = all_preds[sent_idx]
            for i, (token, _) in enumerate(sent):
                pred_tag = idx2tag[preds[i]]
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"[{language}] Test predictions written to {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    languages = ['English', 'Chinese']
    if len(sys.argv) > 1:
        languages = [sys.argv[1]]

    summary = {}
    for lang in languages:
        output_file = os.path.join(data_dir, f'transformer_crf_result_{lang.lower()}.txt')
        model, vocab, tag2idx, idx2tag, char2idx, metrics = train_and_predict(
            lang, data_dir, output_file, device=device
        )
        summary[lang] = metrics

        test_path = os.path.join(data_dir, lang, 'test.txt')
        if os.path.exists(test_path):
            test_output = os.path.join(data_dir, f'transformer_crf_test_result_{lang.lower()}.txt')
            predict_test(model, vocab, tag2idx, idx2tag, char2idx, lang, test_path, test_output, device)

    if len(summary) > 1:
        print()
        print("#" * 64)
        print("#  Transformer+CRF — Overall Summary (micro avg on validation)")
        print("#" * 64)
        print(f"  {'Language':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
        print("  " + "-" * 50)
        for lang, (p, r, f1) in summary.items():
            print(f"  {lang:<12s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}")
        print("#" * 64)
