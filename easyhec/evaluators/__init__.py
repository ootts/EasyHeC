from ..registry import EVALUATORS
from .build import *


def build_evaluators(cfg):
    evaluators = []
    for e in cfg.test.evaluators:
        evaluators.append(EVALUATORS[e](cfg))
    return evaluators
