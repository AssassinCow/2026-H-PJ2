from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]

sorted_labels_chn = [
'O',
'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
, 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
, 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
, 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
, 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
, 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
, 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
, 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

def check(language, gold_path, my_path):
    if language == "English":
        sort_labels = sorted_labels_eng
    else:
        sort_labels = sorted_labels_chn
    y_true = []
    y_pred = []

    def load_tagged_tokens(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.strip().split()
                assert len(parts) == 2, (
                    f"Expected '<token> <tag>' at {path}:{line_no}, got: {line.strip()!r}"
                )
                rows.append((line_no, parts[0], parts[1]))
        return rows

    gold_rows = load_tagged_tokens(gold_path)
    pred_rows = load_tagged_tokens(my_path)
    assert len(gold_rows) == len(pred_rows), (
        f"Token count is not equal: gold={len(gold_rows)}, pred={len(pred_rows)}"
    )
    for (g_line, g_word, g_tag), (m_line, m_word, m_tag) in zip(gold_rows, pred_rows):
        assert g_word == m_word, (
            f"Token mismatch: gold {gold_path}:{g_line}={g_word!r}, "
            f"pred {my_path}:{m_line}={m_word!r}"
        )
        y_true.append(g_tag)
        y_pred.append(m_tag)
    print(metrics.classification_report(
        y_true = y_true, y_pred=y_pred, labels=sort_labels[1:], digits=4
    ))
    return

if __name__ == "__main__":
    check(language = "Chinese", gold_path="data/example_data/example_gold_result.txt", my_path="data/example_data/example_my_result.txt")
