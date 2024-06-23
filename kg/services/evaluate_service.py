# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2019.
"""

"""
from Levenshtein import distance


def evaluate(true_result, pred_result):
    true_id2sample = dict()
    for sample in true_result:
        true_id2sample[sample["id"]] = sample
    pred_id2sample = dict()
    for sample in pred_result:
        pred_id2sample[sample["id"]] = sample
    hit_num = 0
    for sample_id, true_sample in true_id2sample.items():
        if sample_id not in pred_id2sample:
            continue
        pred_sample = pred_id2sample[sample_id]
        if is_hit(true_sample, pred_sample):
            hit_num += 1
    score = hit_num / len(true_id2sample)
    return score


def is_hit(true_sample, pred_sample):
    dist = distance(true_sample["answer"],
                    pred_sample["answer"])
    score = dist / (len(true_sample["answer"]) +
                    len(pred_sample["answer"]))
    if score < 0.1:
        return True
    else:
        return False
