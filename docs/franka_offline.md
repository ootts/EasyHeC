1. Data preparation:

   Prepare images, camera intrinsics, and qposes as the following format:

   ```
   EasyHeC/ # project root
   ├─ data
   │  ├─ franka
   │  │  ├─ example
   │  │  │  ├─ color
   │  │  │     ├─ 000000.png, 000001.png ...
   │  │  │  ├─ qpos
   │  │  │     ├─ 000000.txt, 000001.txt ... (9-dim, last 2 dims are 0, 0)
   │  │  │  ├─ K.txt (3x3 intrinsics)
   ```

   An example is at [assets/franka_offline_example.zip](assets/franka_offline_example.zip).

   During capturing, remember to use as diverse qposes as possible. 

   Use no less than 10 images but no more than 20.

   When capturing each image, let the robot stop to reduce the impact of inaccurate qpos and image synchronization.

1. Annotate masks:

   ```
   python easyhec/utils/prompt_drawer.py --img_paths data/franka/example/color/*png --output_dir data/franka/example/mask
   ```

1. Run easyhec:

   ```bash
   python tools/run_easyhec.py -c configs/franka/example_franka_offline.yaml
   ```

   Before running, check the yaml file to fill in the correct image size, data_path, and init_Tc_c2b. init_Tc_c2b can be obtained by tools/manual_tune_franka_init.py.
