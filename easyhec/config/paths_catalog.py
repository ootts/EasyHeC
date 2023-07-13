import os.path as osp
import os


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('~/Datasets')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)

    @staticmethod
    def get(name: str):
        if name.startswith("xarm7_real"):
            return get_xarm7_real(name)
        elif name.startswith("franka_real"):
            return get_franka_real(name)
        else:
            raise RuntimeError("Dataset not available: {}".format(name))


def get_xarm7_real(name):
    items = name.split("/")[1:]
    data_dir = osp.join("data", "/".join(items))
    return dict(
        factory='XarmRealDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )

def get_franka_real(name):
    items = name.split("/")[1:]
    data_dir = osp.join("data", "/".join(items))
    return dict(
        factory='XarmRealDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )
