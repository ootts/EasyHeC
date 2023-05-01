import datetime
import glob
import os
import os.path as osp
import random
from shutil import copytree, ignore_patterns

import loguru
import numpy as np
import torch
import zarr
from termcolor import colored


def isckpt(fname: str):
    return fname.endswith('.pth')


def red(s: str):
    return colored(s, 'red')


def cyan(s: str):
    return colored(s, 'cyan')


def magenta(s: str):
    return colored(s, 'magenta')


def print_if_enable(*args, sep=' ', end='\n', file=None, enable=True):
    if enable:
        print(*args, sep=sep, end=end, file=file)


def dict_to_str(d: dict, float_precision=2):
    ss = []
    for k, v in d.items():
        if isinstance(v, float):
            if v > 1:
                s = f'{k}={v:.{float_precision}f}'
            else:
                s = f'{k}={v:.{float_precision}e}'
        else:
            s = f'{k}={v}'
        ss.append(s)
    s = ', '.join(ss)
    return s


def zarr_load(store):
    a = zarr.load(store)
    assert a is not None, f'{store} do not exist!'
    return a


class NotCheckedError(RuntimeError):
    pass


def archive_runs(output_dir):
    from easyhec.utils.comm import get_rank, synchronize
    if get_rank() == 0:
        timestamp_file = osp.join(output_dir, 'timestamp.txt')
        if osp.exists(timestamp_file):
            with open(timestamp_file) as f:
                timestamp = 'start_time' + f.read()
        else:
            timestamp = 'archive_time_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        archive_dir = osp.join(output_dir, 'archives', timestamp)
        os.makedirs(archive_dir)
        for file in os.listdir(output_dir):
            if file != 'archives':
                os.system(f'mv {osp.join(output_dir, file)} {archive_dir}')
        # delete event file
        event_files = glob.glob(osp.join(archive_dir, 'events*'))
        if len(event_files) > 0 and osp.getsize(event_files[0]) < 1000:
            print(f'deleting {event_files[0]}')
            os.system(f'rm {event_files[0]}')
        with open(osp.join(output_dir, 'timestamp.txt'), 'w') as f:
            f.write(str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    synchronize()


def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))


def make_source_code_snapshot(log_dir):
    if not osp.exists(osp.join(log_dir, 'source')):
        loguru.logger.info("Backuping source....")
        os.makedirs(osp.join(log_dir, 'source'))
        cmd = f'rsync -r --include crc/data --include crc/modeling/models --include *'
        exclusions = ['__pycache__', 'data', "logs", "models", "*.egg-info", ".vscode", "*.so", "*.a",
                      ".ipynb_checkpoints", "build", "bin", "*.ply", "eigen", "pybind11", "*.npy", "*.pth", ".git",
                      "debug", "dbg", "tmp","assets"]
        exclusion = ' '.join(['--exclude ' + a for a in exclusions])
        cmd = cmd + " " + exclusion
        cmd = cmd + f" . {log_dir}/source/"
        cmd = cmd + " --info=progress2 --no-i-r"
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError("copy files failed!!!!")
        loguru.logger.info("Backuping source done!")


def deterministic():
    np.random.seed(0)
    random.seed(0)
    # torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(0)


def number_of_monitors():
    import screeninfo
    try:
        monitors = screeninfo.get_monitors()
        return len(monitors)
    except Exception as ex:
        return 0

# def main():
#     print(number_of_monitors())
#
#
# if __name__ == '__main__':
#     main()
