from easyhec.modeling.models.rb_solve.rb_solver import RBSolver
from easyhec.modeling.models.rb_solve.space_explorer import SpaceExplorer

_META_ARCHITECTURES = {
    'RBSolver': RBSolver,
    "SpaceExplorer": SpaceExplorer,
}


def build_model(cfg):
    print("building model...", end='\r')
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    model = meta_arch(cfg)
    return model
