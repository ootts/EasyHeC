from ..registry import VISUALIZERS


def build_visualizer(cfg):
    return VISUALIZERS[cfg.test.visualizer](cfg)
