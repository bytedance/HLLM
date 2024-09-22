# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

from enum import Enum


class InputType(Enum):

    SEQ = 1
    PAIR = 2
    AUGSEQ = 3


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """
    RANKING = 1
    VALUE = 2
