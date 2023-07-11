# Support other cameras

​	To support other cameras, you need to customize the data capture pipeline similar to [here](../easyhec/utils/realsense_api.py) and modify the code [here](../easyhec/trainer/rbsolve_iter.py#L214). You need to perform undistortion by yourself.

# Support other robot arms

​	To support other robot arms, you need to do the following things.
​	1) Since we don't have other types of robot arms in our lab, you need to a) tune the textures of the robot model, b) define the [mesh_mapping](../easyhec/structures/xarm_mapping.py), c) modify the control code, and d) write the [render_api](../easyhec/utils/render_api.py#L136) for your robot.
​	2) Obtain the segmentation network.
​	3) Initialize the camera pose.

## 1. Obtain Segmentation mask

We provide two approaches to obtain the segmentation mask: 1) Train a PointRend, or 2) Manually label by [SAM](https://segment-anything.com/).

### Train a PointRend

1) Render data in SAPIEN

​		a) The main script for the rendering is at ``tools/simulate/gen_data_for_mask_training.py``.

​		b) Split the generated data to train/val split with ``tools/simulate/split_mask_training_data.py``.

2. Train PointRend using synthetic data

   a) Register the dataset at [builtin.py](../third_party/detectron2/detectron2/data/datasets/builtin.py#L321).

   b) Link the generated dataset.

   ```bash
   ln -s data/xarm7/simulate/mask_data third_party/detectron2/datasets/xarm7
   ```

   c) Train the PointRend

   ```bash
   cd third_party/detectron2/projects/PointRend/
   python train_net.py -c configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm7.yaml
   ```

3. Finetune the model on some real data (Optional, but recommended)

   a) Capture real data and organize as the example data. Download the example data using the following command.

   ```bash
   cd data/xarm7/simulate/mask_data/
   gdown 1FhVb-Ba154x8jdQxXiq_520EXnKkFJhN
   unzip xarm7_finetune.zip && rm xarm7_finetune.zip
   cd ../../../../
   ```

   b) Register the dataset at [builtin.py](../third_party/detectron2/detectron2/data/datasets/builtin.py#L321).

   c) Train the PointRend

   ```bash
   python -c train_net.py -c configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm7_finetune.yaml
   ```

### Manually label by SAM.

1. Setup [SAM](https://github.com/facebookresearch/segment-anything) following the instructions [here](https://github.com/facebookresearch/segment-anything#installation), and download the checkpoint with the following command.

```bash
mkdir -p models/sam/ && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam/sam_vit_h_4b8939.pth
```

2. Run EasyHeC with the following command. The prompt drawer released on 2023.7.11 allows you to draw multiple boxes and take the union of them as the robot mask. Press "z" to undo, "p" to switch to the point prompt mode, and "b" to switch to the bounding box prompt mode. Although the point prompt is supported, we recommend using the bounding box prompt since it's more stable than the point prompt.

```bash
python tools/run_easyhec.py -c configs/xarm7/example_sam.yaml
```

## 2. Initialize the camera pose

To initialize the camera pose for the the robot other than XArm7, we provide the following approaches:

1) Train a PVNet as describe in our paper. This is time-consuming but generalize well to different camera poses.
2) Initialize with the results of other methods, such as traditional marker-based methods. This is convenient if you already have implementations for other methods.
3) Manually initialize the camera pose by tuning.
4) Initialize the camera pose with last result. This is convenient if the camera poses are roughly the same each time.

Here, we provide example for training the PVNet.

1. Generate data in SAPIEN.

   ```bash
   python tools/simulate/gen_data_for_pvnet.py -c configs/xarm7/simulate/pvnet_data.yaml
   python tools/simulate/convert_pvnet_data_to_pvnet_format.py -c configs/xarm7/simulate/pvnet_data.yaml
   cd third_party/pvnet/
   python run.py --type custom custom.data_dir ../../data/xarm7/simulate/pvnet_data/pvnet_format/train
   python run.py --type custom custom.data_dir ../../data/xarm7/simulate/pvnet_data/pvnet_format/test
   ```

2. Register the dataset at [here](../third_party/pvnet/lib/datasets/dataset_catalog.py).

3. Train PVNet.

   ```bash
   python train_net.py -c third_party/pvnet/configs/xarm7/10k.yaml
   ```

   
