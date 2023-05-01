import os.path as osp

import numpy as np

from lib.config import cfg


def run_demo():
    from lib.visualizers import make_visualizer
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image

    torch.manual_seed(0)
    meta = np.load(cfg.demo_meta_path, allow_pickle=True).item()
    if cfg.custom.K == []:
        meta['K'] = np.array(meta['K'])
    else:
        print("using K from command line")
        meta['K'] = np.array(cfg.custom.K)

    demo_images = glob.glob(osp.join(cfg.demo_dir, f'{cfg.demo_pattern}'))
    demo_images = sorted(demo_images)[:1]

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    np.set_printoptions(suppress=True, precision=4)
    # for i, path in enumerate(tqdm.tqdm(demo_images)):
    demo_image_full = np.array(Image.open(demo_images[0])).astype(np.float32)
    demo_image = demo_image_full
    inp = (((demo_image / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
    inp = torch.Tensor(inp[None]).cuda()
    with torch.no_grad():
        output = network(inp)
    output['global_step'] = 0
    output['demo_image_full'] = demo_image_full / 255.0
    pose_pred = visualizer.visualize_demo(output, inp, meta)
    pose_pred = np.concatenate((pose_pred, np.array([[0, 0, 0, 1]])), axis=0)
    print(np.array2string(pose_pred, separator=', '))
    return pose_pred


if __name__ == '__main__':
    run_demo()
