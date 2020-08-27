""" compute logit for QA model """
import logging
import math
import collections
import re
import string
from itertools import groupby

from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ('compute_logits', 'compute_gold_answer', 'get_evaluation')


# For evaluation

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(gold_answers, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    has_answer_qids, has_no_answer_qids = [], []
    for qas_id, answer in gold_answers.items():
        gold_answers = [a for a in answer if normalize_answer(a)]

        if len(gold_answers) == 0:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]
            has_no_answer_qids.append(qas_id)
        else:
            has_answer_qids.append(qas_id)

        if qas_id not in preds.keys():
            print("Missing prediction for %s" % qas_id)
            continue

        exact_scores[qas_id] = max(compute_exact(a, preds[qas_id]) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, preds[qas_id]) for a in gold_answers)

    return exact_scores, f1_scores, has_answer_qids, has_no_answer_qids


def get_evaluation(gold_answers, preds):
    exact_scores, f1_scores, has_answer_qids, has_no_answer_qids = get_raw_scores(gold_answers, preds)
    total = len(exact_scores)
    metrics = collections.OrderedDict([
        ("exact", 100.0 * sum(exact_scores.values()) / total),
        ("f1", 100.0 * sum(f1_scores.values()) / total),
        ("total", total)
    ])
    if len(has_answer_qids) != 0:
        metrics.update(collections.OrderedDict([
            ("HasAns_exact", 100.0 * sum(exact_scores[k] for k in has_answer_qids) / len(has_answer_qids)),
            ("HasAns_f1", 100.0 * sum(f1_scores[k] for k in has_answer_qids) / len(has_answer_qids)),
            ("HasAns_total", len(has_answer_qids))
        ]))
    if len(has_no_answer_qids) != 0:
        metrics.update(collections.OrderedDict([
            ("NoAns_exact", 100.0 * sum(exact_scores[k] for k in has_no_answer_qids) / len(has_no_answer_qids)),
            ("NoAns_f1", 100.0 * sum(f1_scores[k] for k in has_no_answer_qids) / len(has_no_answer_qids)),
            ("NoAns_total", len(has_no_answer_qids)),
            ]))
    return metrics


# For logit computation

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_gold_answer(
        start_positions,  # (batch, )
        end_positions,  # (batch, )
        question_id,  # (batch, )
        input_ids,  # (batch, max_length)
        tokenizer):
    cls_index = input_ids[0].index(tokenizer.cls_token_id)
    all_gold_answer = {}
    for _start_positions, _end_positions, _question_id, _input_ids in \
            zip(start_positions, end_positions, question_id, input_ids):

        if _start_positions == cls_index and _end_positions == cls_index:
            correct_answers = ''
        else:
            correct_answers = tokenizer.decode(_input_ids[_start_positions: _end_positions], skip_special_tokens=True)

        # keep only unique answer
        if _question_id not in all_gold_answer.keys():
            all_gold_answer[_question_id] = [correct_answers]
        elif correct_answers not in all_gold_answer[_question_id]:
            all_gold_answer[_question_id] += [correct_answers]

    return all_gold_answer


def compute_logits(
        start_logits,  # (batch, max_length)
        end_logits,  # (batch, max_length)
        question_id,  # (batch, )
        length,  # (batch, )
        doc_offset,  # (batch, )
        token_is_max_context,  # (batch, )
        input_ids,  # (batch, max_length)
        tokenizer,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        version_2_with_negative=True,
        retrun_best_only: bool = True):
    """Write final predictions to the json file and log-odds of null if needed."""
    cls_index = input_ids[0].index(tokenizer.cls_token_id)
    all_nbest = {}

    def mask_list(_list, _q_id):
        assert len(_list) == len(question_id)
        return [i for i, m in zip(_list, question_id) if m == _q_id]

    unique_question_ids = [i[0] for i in groupby(question_id)]
    for q_id in tqdm(unique_question_ids):  # loop over unique questions
        _start_logits = mask_list(start_logits, q_id)
        _end_logits = mask_list(end_logits, q_id)
        _length = mask_list(length, q_id)
        _doc_offset = mask_list(doc_offset, q_id)
        _token_is_max_context = mask_list(token_is_max_context, q_id)
        _input_ids = mask_list(input_ids, q_id)

        prelim_predictions = []
        seen_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = None  # large and positive
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        # loop within a unique question (over multiple context truncations)
        for __start_logits, __end_logits, __length, __doc_offset, __token_is_max_context, __input_ids in zip(
            _start_logits, _end_logits, _length, _doc_offset, _token_is_max_context, _input_ids):

            start_indexes = _get_best_indexes(__start_logits, n_best_size)
            end_indexes = _get_best_indexes(__end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative and (
                    score_null is None or __start_logits[cls_index] + __end_logits[cls_index] < score_null):
                null_start_logit = __start_logits[cls_index]
                null_end_logit = __end_logits[cls_index]
                score_null = null_start_logit + null_end_logit

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if end_index < start_index:
                        continue
                    # answer can't be in the question
                    if start_index < __doc_offset:
                        continue
                    # answer can't be out of total input
                    if end_index > __doc_offset + __length:
                        continue
                    if not __token_is_max_context.get(start_index, False):
                        continue

                    if end_index - start_index + 1 > max_answer_length:
                        continue
                    answer = tokenizer.decode(__input_ids[start_index: end_index], skip_special_tokens=True)
                    if answer in seen_predictions:
                        continue
                    prelim_predictions.append({
                        # "start_index": start_index, "end_index": end_index,
                        "start_logit": __start_logits[start_index], "end_logit": __end_logits[end_index],
                        "total_scores": __start_logits[start_index] + __end_logits[end_index],
                        "text": answer
                    })
                    seen_predictions.append(answer)
        if version_2_with_negative:
            prelim_predictions.append({
                # "start_index": cls_index, "end_index": cls_index,
                "start_logit": null_start_logit, "end_logit": null_end_logit,
                "total_scores": null_start_logit + null_end_logit,
                "text": ""
            })
            seen_predictions.append("")
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x['total_scores']), reverse=True)
        nbest = prelim_predictions[:min(len(prelim_predictions), n_best_size)]

        assert len(nbest) > 0, "No valid predictions"

        probs = _compute_softmax([entry['total_scores'] for entry in nbest])
        for n, entry in enumerate(nbest):
            entry['probability'] = probs[n]

        if retrun_best_only:
            all_nbest[q_id] = nbest[0]['text']
        else:
            all_nbest[q_id] = nbest
    return all_nbest

