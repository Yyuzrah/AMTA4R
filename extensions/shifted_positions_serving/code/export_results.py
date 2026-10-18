"""Copy the two tables and the per-model cards that analyze.py writes to <base>/analysis/ into the released result
folder. Cell values are copied as text, unchanged; some table columns get plainer names (COLUMNS). In the cards, the
line with the downloaded commit becomes a line with the hub id only, the admission line is dropped, and the table
header and one recommendation text are reworded (CARD_TEXT).
Usage: export_results.py <base> [out dir, default <base>/results]"""
import csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import ORDER

COLUMNS = {'class': 'position_encoding', 'c_to_w': 'n_correct_to_wrong', 'w_to_c': 'n_wrong_to_correct',
           'material': 'abs_delta_ge_0.02_and_ci_excludes_0', 'frac_logit_changed': 'frac_logit_changed_gt_1e-3',
           'logit_maxabs_median': 'median_max_abs_logit_diff', 'logit_maxabs_max': 'max_max_abs_logit_diff',
           'kl_median': 'median_first_token_kl'}
CELLS_HEADER = ['model', 'class', 'scenario', 'task', 'row', 'n', 'acc_ref', 'acc_row', 'delta', 'ci_lo', 'ci_hi',
                'answer_flip_rate', 'c_to_w', 'w_to_c', 'first_token_flip_rate', 'logit_maxabs_median',
                'logit_maxabs_max', 'kl_median']
POOLED_HEADER = ['model', 'class', 'scenario', 'row', 'n', 'tasks', 'acc_ref', 'acc_row', 'delta', 'ci_lo', 'ci_hi',
                 'material', 'answer_flip_rate', 'first_token_flip_rate', 'frac_logit_changed']
HUB_LINE = re.compile(r'^- hub `([^`]+)` rev `[^`]*`; licence: (.*)$')
CARD_TEXT = [
    ('| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | '
     'median max|dlogit| |',
     '| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | '
     'median max abs logit diff |'),
    ('- reuse a prefix KV cache computed at a different (block) offset:',
     '- reuse a prefix KV cache computed with all position indices shifted (block):'),
]


def copy_table(src, dst, header):
    rows = list(csv.reader(open(src, newline='')))
    assert rows[0] == header, (src, rows[0])
    with open(dst, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([COLUMNS.get(c, c) for c in rows[0]])
        w.writerows(rows[1:])
    return len(rows) - 1


def copy_card(src, dst, tag):
    out, n_hub, n_adm = [], 0, 0
    lines = open(src).read().split('\n')
    title = f'# Answers under shifted position indices: {tag}'
    assert lines[0] == title, (src, lines[0])
    for i, line in enumerate(lines):
        if i == 0:
            out.append(title)
            continue
        m = HUB_LINE.match(line)
        if m:
            out.append(f'- Hugging Face Hub model `{m.group(1)}`; license: {m.group(2)}')
            n_hub += 1
            continue
        if line.startswith('- admitted to the main checks:'):
            n_adm += 1
            continue
        for old, new in CARD_TEXT:
            if line.startswith(old):
                line = new + line[len(old):]
        out.append(line)
    assert n_hub == 1 and n_adm == 1, (src, n_hub, n_adm)
    open(dst, 'w').write('\n'.join(out))


def main():
    base = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f'{base}/results'
    src = f'{base}/analysis'
    os.makedirs(f'{out}/cards', exist_ok=True)
    n1 = copy_table(f'{src}/table_cells.csv', f'{out}/table_cells.csv', CELLS_HEADER)
    n2 = copy_table(f'{src}/table_pooled.csv', f'{out}/table_pooled.csv', POOLED_HEADER)
    n3 = 0
    for tag in ORDER:
        p = f'{src}/cards/{tag}.md'
        if os.path.exists(p):
            copy_card(p, f'{out}/cards/{tag}.md', tag)
            n3 += 1
    print(f'{out}: table_cells.csv {n1} rows, table_pooled.csv {n2} rows, {n3} cards')
    return 0


if __name__ == '__main__':
    sys.exit(main())
