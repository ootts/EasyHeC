from dl_ext import AverageMeter
from dl_ext.timer import EvalTime

from lib.config import cfg


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    # vis_dir = osp.join(cfg.model_dir, cfg.demo_path.replace("/", "_"))
    # os.makedirs(vis_dir, exist_ok=True)
    am = AverageMeter()
    evaltime = EvalTime()
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        evaltime("begin")
        with torch.no_grad():
            output = network(batch['inp'])
        evaltime("end")
        visualizer.visualize(output, batch, show=True)
        # plt.savefig(osp.join(vis_dir, str(i) + ".png"))


if __name__ == '__main__':
    run_visualize()
