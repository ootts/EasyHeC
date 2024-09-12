import torch
import os
import io
# import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from loguru import logger
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

from easyhec.trainer.base import BaseTrainer
from easyhec.trainer.utils import *
from easyhec.utils import plt_utils


class RBSolverTrainer(BaseTrainer):
    def train(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        metric_ams = {}
        bar = tqdm(self.train_dl, leave=False) if is_main_process() else self.train_dl
        begin = time.time()
        for batchid, batch in enumerate(bar):
            self.optimizer.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = batchid
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            # record and plot loss and metrics
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if 'metrics' in output:
                for k, v in output['metrics'].items():
                    reduced_s = reduce_loss(v)
                    metrics[k] = reduced_s
            if is_main_process() and epoch % self.cfg.solver.log_interval == 0:
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                if self.global_steps % 100 == 0:
                    self.image_grid_on_tb_writer(output['rendered_masks'], self.tb_writer,
                                                 'train/rendered_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['ref_masks'], self.tb_writer,
                                                 'train/ref_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['error_maps'], self.tb_writer,
                                                 "train/error_maps", self.global_steps)
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
                for k, v in metrics.items():
                    if k not in metric_ams.keys():
                        metric_ams[k] = AverageMeter()
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()

    def image_grid_on_tb_writer(self, images, tb_writer, tag, global_step):
        plt_utils.image_grid(images, show=False)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        tb_writer.add_image(tag, np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), global_step)
        plt.close("all")

    def save(self, epoch):
        if self.save_mode == 'epoch':
            name = os.path.join(self.output_dir, 'model_epoch_%06d.pth' % epoch)
        else:
            name = os.path.join(self.output_dir, 'model_iteration_%06d.pth' % self.global_steps)
        net_sd = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        # if hasattr(self,"cfg.solver.pop_verts_faces") and self.cfg.solver.pop_verts_faces:
        #     net_sd = {k: v for k, v in net_sd.items() if 'vert' not in k and 'faces' not in k}
        # if self.cfg.solver.compress_history_ops and "history_ops" in net_sd:
        #     keep = (net_sd["history_ops"] != 0).any(dim=1)
        #     net_sd["history_ops"] = net_sd["history_ops"][keep]
        d = {'model': net_sd,
             'epoch': epoch,
             'best_val_loss': self.best_val_loss,
             'global_steps': self.global_steps}
        if self.cfg.solver.save_optimizer:
            d['optimizer'] = self.optimizer.state_dict()
        if self.cfg.solver.save_scheduler:
            d['scheduler'] = self.scheduler.state_dict()
        torch.save(d, name)
