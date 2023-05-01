import sys

import numpy as np


def pvnet_init_pose(image_path,demo_meta_path):
    sys.path.append("third_party/pvnet")
    # from lib.config import cfg
    from lib.visualizers import make_visualizer
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from PIL import Image

    torch.manual_seed(0)
    meta = np.load(cfg.demo_meta_path, allow_pickle=True).item()
    meta['K'] = np.array(meta['K'])

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    np.set_printoptions(suppress=True, precision=4)
    demo_image = np.array(Image.open(image_path)).astype(np.float32)
    inp = (((demo_image / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
    inp = torch.Tensor(inp[None]).cuda()
    with torch.no_grad():
        output = network(inp)
    pose_pred = visualizer.visualize_demo(output, inp, meta)
    pose_pred = np.concatenate((pose_pred, np.array([[0, 0, 0, 1]])), axis=0)
    print(np.array2string(pose_pred, separator=', '))
    return pose_pred
