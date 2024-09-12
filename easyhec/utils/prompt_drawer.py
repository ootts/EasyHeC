import hashlib
import os
import os.path as osp
import time

import imageio
import loguru
import torch
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

import numpy as np
import cv2
from enum import Enum

from dl_ext.timer import EvalTime

from easyhec.utils import plt_utils


class DrawingMode(Enum):
    Box = 0
    Point = 1


class PromptDrawer(object):
    def __init__(self, window_name="Prompt Drawer", screen_scale=1.0, sam_checkpoint=""):
        self.window_name = window_name  # Name for our window
        self.reset()
        self.screen_scale = screen_scale * 1.2
        self.screen_scale = screen_scale

        ################  ↓ Step : build SAM  ↓  ##############
        device = "cuda"
        model_type = "default"

        import sys

        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def reset(self):
        self.done = False
        self.drawing = False
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.box = np.zeros([4], dtype=np.float32)
        self.points = np.empty((0, 2))  # List of points defining our polygon
        self.labels = np.empty([0], dtype=int)
        self.mask = None
        self.mode = DrawingMode.Box
        self.boxes = np.zeros([0, 4], dtype=np.float32)
        self.box_labels = np.empty([0], dtype=int)

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return
        if self.mode == DrawingMode.Box:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    self.box_labels = np.hstack([self.box_labels, 0])
                else:
                    self.box_labels = np.hstack([self.box_labels, 1])
                self.boxes = np.vstack([self.boxes, [x, y, x, y]])
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.boxes[-1, 2] = x
                self.boxes[-1, 3] = y
                self.boxes[-1, 0], self.boxes[-1, 2] = min(self.boxes[-1, 0], self.boxes[-1, 2]), max(self.boxes[-1, 0],
                                                                                                      self.boxes[-1, 2])
                self.boxes[-1, 1], self.boxes[-1, 3] = min(self.boxes[-1, 1], self.boxes[-1, 3]), max(self.boxes[-1, 1],
                                                                                                      self.boxes[-1, 3])
                self.detect()
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.boxes[-1, 2] = x
                    self.boxes[-1, 3] = y
        elif self.mode == DrawingMode.Point:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points = np.vstack([self.points, [x, y]])
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    label = 0
                else:
                    label = 1
                self.labels = np.hstack([self.labels, label])
                self.detect()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points = np.vstack([self.points, [x, y]])
                self.labels = np.hstack([self.labels, 1])
                self.detect()

    def detect(self):
        if len(self.points) != 0:
            input_point = self.points / self.ratio
            input_label = self.labels.astype(int)
        else:
            input_point = None
            input_label = None
        final_mask = None
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            box_label = self.box_labels[i]
            if np.all(box == 0):
                box = None
            else:
                box = box / self.ratio
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                # mask_input=
                box=box,
                multimask_output=True,
            )
            maxidx = np.argmax(scores)
            mask = masks[maxidx]
            if final_mask is None:
                final_mask = mask.copy()
            else:
                if box_label == 0:
                    final_mask = np.logical_and(final_mask, ~mask)
                else:
                    final_mask = np.logical_or(final_mask, mask)
        if final_mask is not None:
            self.mask = final_mask.copy()
        elif self.mask is not None:
            self.mask = np.zeros_like(self.mask)

    def run(self, rgb):
        self.rgb = rgb
        self.predictor.set_image(rgb)
        image_to_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_h, image_w = image_to_show.shape[:2]

        if not hasattr(self, "ratio"):
            output = subprocess.check_output(["xrandr"]).decode("utf-8")
            current_mode = [line for line in output.splitlines() if "*" in line][0]
            screen_width, screen_height = [int(x) for x in current_mode.split()[0].split("x")]
            scale = self.screen_scale
            screen_w = int(screen_width / scale)
            screen_h = int(screen_height / scale)

            ratio = min(screen_w / image_w, screen_h / image_h)
            self.ratio = ratio
        target_size = (int(image_w * self.ratio), int(image_h * self.ratio))
        image_to_show = cv2.resize(image_to_show, target_size)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image_to_show)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            tmp = image_to_show.copy()
            tmp = cv2.circle(tmp, self.current, radius=2,
                             color=(0, 0, 255),
                             thickness=-1)
            for box, box_label in zip(self.boxes, self.box_labels):
                color = (0, 255, 0) if box_label == 1 else (0, 0, 255)
                cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            if self.points.shape[0] > 0:
                for ptidx, pt in enumerate(self.points):
                    color = (0, 255, 0) if self.labels[ptidx] == 1 else (0, 0, 255)
                    tmp = cv2.circle(tmp, (int(pt[0]), int(pt[1])), radius=5,
                                     color=color,
                                     thickness=-1)
            if self.mask is not None:
                m = self.mask
                mask_to_show = cv2.resize(m.astype(np.uint8), target_size).astype(bool)
                tmp = plt_utils.vis_mask(tmp, mask_to_show.astype(np.uint8), color=[0, 255, 0], alpha=0.5).astype(np.uint8)
            cv2.imshow(self.window_name, tmp)
            waittime = 50
            key = cv2.waitKey(waittime)
            if key == 13:  # enter
                self.mask = np.zeros([image_h, image_w], dtype=bool)
                self.done = True
            if key == 27:  # ESC hit
                self.done = True
            elif key == ord('r'):
                print("Reset")
                self.reset()
            elif key == ord('p'):
                print("Switch to point mode")
                self.mode = DrawingMode.Point
            elif key == ord('b'):
                print("Switch to box mode")
                self.mode = DrawingMode.Box
            elif key == ord('z'):
                print("Undo")
                if self.mode == DrawingMode.Point and len(self.points) > 0:
                    self.points = self.points[:-1]
                    self.labels = self.labels[:-1]
                    self.detect()
                elif self.mode == DrawingMode.Box and len(self.boxes) > 0:
                    self.boxes = self.boxes[:-1]
                    self.box_labels = self.box_labels[:-1]
                    self.detect()
        cv2.destroyWindow(self.window_name)
        # del self.predictor
        # torch.cuda.empty_cache()
        return None, None, self.mask

    def close(self):
        del self.predictor
        torch.cuda.empty_cache()
        return None, None, self.mask


# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dont_save", action="store_true")
    args = parser.parse_args()

    print('Usage: drag mouse to draw bounding boxes. Ctrl+mouse to draw negative bounding boxes. Press "p" to switch to point mode, "b" to switch to box mode, "z" to undo, "r" to reset, "ESC" to finish.')

    i = 0
    sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth"
    if not osp.exists(sam_checkpoint) or hashlib.md5(open(sam_checkpoint, 'rb').read()).hexdigest() != "4b8939a88964f0f4ff5f5b2642c598a6":
        os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P third_party/segment_anything')
    drawer = PromptDrawer(screen_scale=3.0, sam_checkpoint=sam_checkpoint)
    while i < len(args.img_paths):
        img_path = args.img_paths[i]
        rgb = imageio.imread_v2(img_path)[..., :3]
        _, _, mask = drawer.run(rgb)
        if mask is not None:
            if not args.dont_save:
                if args.output_dir == "":
                    out_path = img_path[:-4] + "mask" + img_path[-4:]
                else:
                    out_path = osp.join(args.output_dir, osp.basename(img_path[:-4] + '.png'))
                    os.makedirs(args.output_dir, exist_ok=True)
                imageio.imwrite(out_path, ((mask > 0) * 255).astype(np.uint8))
        i += 1
        drawer.reset()
    # rgb = imageio.imread_v2(args.img_path)
    # mask = PromptDrawer(screen_scale=3.0).run(rgb)
    # plt.imshow(mask)
    # plt.show()

    # imageio.imwrite(args.img_path[:-4] + "mask" + args.img_path[-4:], ((mask > 0) * 255).astype(np.uint8))
