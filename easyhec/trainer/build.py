from easyhec.registry import TRAINERS
from easyhec.trainer.base import BaseTrainer
from easyhec.trainer.rbsolver import RBSolverTrainer
from easyhec.trainer.rbsolve_iter import RBSolverIterTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('rbsolver')
def build_rbsolver_trainer(cfg):
    return RBSolverTrainer(cfg)


@TRAINERS.register('rbsolver_iter')
def build_rbsolveriter_trainer(cfg):
    return RBSolverIterTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
