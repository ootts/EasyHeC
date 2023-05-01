from argparse import Namespace
from typing import Union
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.data.transforms import AugmentationList
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import Visualizer, ColorMode
from easydict import EasyDict


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class PointRendApiHelper:
    _augmentations = None
    _model = None
    set = False
    _config_file = None
    _model_weight = None

    @staticmethod
    def get_augmentations():
        return PointRendApiHelper._augmentations

    @staticmethod
    def get_model():
        return PointRendApiHelper._model

    @staticmethod
    def setup_helper(config_file, model_weight):
        if PointRendApiHelper.set and config_file == PointRendApiHelper._config_file and model_weight == PointRendApiHelper._model_weight:
            return
        PointRendApiHelper._config_file = config_file
        PointRendApiHelper._model_weight = model_weight
        # args = default_argument_parser().parse_args()
        args = EasyDict()
        args.config_file = config_file
        args.resume = False
        args.opts = ["MODEL.WEIGHTS", model_weight, "DATALOADER.NUM_WORKERS", "0"]

        cfg = setup(args)

        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()
        PointRendApiHelper._model = model
        augmentations = detection_utils.build_augmentation(cfg, is_train=False)
        augmentations = AugmentationList(augmentations)
        PointRendApiHelper._augmentations = augmentations
        PointRendApiHelper.set = True


def pointrend_api(config_file, model_weight, image: Union[str, np.ndarray]):
    """
    :param config_file: path to config file
    :param model_weight: path to model weight
    :param image: path to image if str, or RGB image if np.ndarray
    :return: binary mask: numpy array; shape H,W; values 0,1; type np.uint8
    """
    PointRendApiHelper.setup_helper(config_file, model_weight)
    augmentations = PointRendApiHelper.get_augmentations()
    model = PointRendApiHelper.get_model()
    if isinstance(image, str):
        image_path = image
        image = detection_utils.read_image(image, format="BGR")
    else:
        image = image.copy()
        image = image[:, :, ::-1]
        image_path = "dummy.png"
    dataset_dict = {"file_name": image_path, "image_id": 0, "height": image.shape[0],
                    "width": image.shape[1]}
    aug_input = T.AugInput(image.copy(), sem_seg=None)
    augmentations(aug_input)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1)))
    inputs = [dataset_dict]
    outputs = model(inputs)
    pred_binary_mask = outputs[0]["instances"].to("cpu").get_fields()['pred_masks'].sum(0).clamp(max=1).bool()
    pred_binary_mask = pred_binary_mask.numpy().astype(np.uint8)
    return pred_binary_mask


def main():
    config_file = "/home/linghao/PycharmProjects/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm.yaml"
    model_weight = "/home/linghao/PycharmProjects/detectron2/projects/PointRend/output/model_0099999.pth"
    image_path = "/home/linghao/PycharmProjects/cam_robot_calib/data/sim_for_hec/space_explore_from_manual3_topk_it00_nostep_rt/color/000000.png"
    image = imageio.imread_v2(image_path)

    pred_binary_mask = pointrend_api(config_file, model_weight, image)

    v = Visualizer(image,
                   metadata=dict(thing_classes=["xarm"]),
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_binary_mask(pred_binary_mask.numpy().astype(np.uint8),
                             color=[0, 1, 0])
    plt.imshow(out.get_image())
    plt.show()


if __name__ == "__main__":
    main()
