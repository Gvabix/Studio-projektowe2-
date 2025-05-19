import re
import ast
from collections import Counter

def simplify_pos(pos_tag):
    """
    Simplify POS tags to a smaller set of categories.
    """
    if pos_tag.startswith('NN'):
        return 'NN'
    elif pos_tag.startswith('VB'):
        return 'VB'
    elif pos_tag.startswith('JJ'):
        return 'JJ'
    elif pos_tag.startswith('RB'):
        return 'RB'
    elif pos_tag == 'DT':
        return 'DT'
    elif pos_tag == 'IN':
        return 'IN'
    elif pos_tag == 'PRP':
        return 'PRP'
    return None


def clean_token(token):
    """
    Handle escaped string formatting and potential byte-string literals.
    """
    try:
        if token.startswith(("b'", 'b"')):
            parsed = ast.literal_eval(token)
            return parsed.decode('utf-8') if isinstance(parsed, bytes) else parsed
        return token
    except Exception:
        return token


def extract_features_from_conll(conll_path):
    """
    Extract token and POS tag statistics from a .conll file.
    """
    pos_counts = {}
    total_tokens = 0
    total_token_len = 0

    with open(conll_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 5:
                continue

            token = parts[1]
            pos_tag = clean_token(parts[4])

            total_tokens += 1
            total_token_len += len(token)

            simplified = simplify_pos(pos_tag)
            if simplified:
                pos_counts[simplified] = pos_counts.get(simplified, 0) + 1

    avg_token_len = total_token_len / total_tokens if total_tokens else 0
    pos_tags = ['NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'PRP']
    pos_vector = [pos_counts.get(tag, 0) / total_tokens if total_tokens else 0 for tag in pos_tags]

    return [total_tokens, avg_token_len] + pos_vector


def extract_features_from_brackets(brackets_path):
    """
    Extract structural features and relation frequencies from a brackets file.
    """
    span_count = 0
    nucleus_count = 0
    satellite_count = 0
    relation_counter = Counter()
    unique_ids = set()

    pattern = re.compile(r"\(\((\d+), (\d+)\), '(\w+)', '(\w+)'\)")

    with open(brackets_path, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                start, end, role, relation = match.groups()
                start, end = int(start), int(end)
                unique_ids.update([start, end])

                if start != end:
                    span_count += 1

                if role == 'Nucleus':
                    nucleus_count += 1
                elif role == 'Satellite':
                    satellite_count += 1

                relation_counter[relation] += 1

    total_nodes = nucleus_count + satellite_count
    return {
        'num_edus': len(unique_ids),
        'num_spans': span_count,
        'num_nucleus': nucleus_count,
        'num_satellite': satellite_count,
        'ratio_nucleus': nucleus_count / total_nodes if total_nodes else 0,
        'ratio_satellite': satellite_count / total_nodes if total_nodes else 0,
        'relation_counts': dict(relation_counter)
    }


def extract_all_features(conll_path, brackets_path):
    """
    Combines all features from different sources into a single flat vector.
    """
    conll_features = extract_features_from_conll(conll_path)
    bracket_features = extract_features_from_brackets(brackets_path)
    return conll_features + list(bracket_features.values())


def separate_dict_features(dict_list):
    """
    Normalize list of dicts: fill missing keys with 0 and return consistent ordering.
    """
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())

    filled_dicts = []
    for d in dict_list:
        filled = {key: d.get(key, 0) for key in all_keys}
        filled_dicts.append(filled)

    return filled_dicts, sorted(all_keys)
