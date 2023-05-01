"""
Manually select frames from demo video to remove too many static frames.
"""
import imageio
import os.path as osp
import glob


def main():
    # duck
    result_dir = "data/model/pvnet/custom_kinect_2022_0801_4_duck_ring24000_and_real_trim0.04/demo_0828_003/-1_icp_smooth"
    frames = [
        *list(range(42, 69)),
        *list(range(167, 171)),
        *list(range(184, 191)),
        *list(range(205, 212)),
        *list(range(235, 237)),
        *list(range(292, 320)),
        *list(range(344, 364)),
    ]
    # box
    result_dir = "data/model/pvnet/custom_kinect_2022_0801_4_box_ring24000_and_real_trim0.04/demo_0828_007/-1_icp_smooth"
    frames = [
        *list(range(21, 57)),
        *list(range(83, 113)),
        *list(range(163, 166)),
        *list(range(184, 191)),
        *list(range(213, 218)),
        *list(range(269, 300)),
    ]
    # cat
    # result_dir = "data/model/pvnet/custom_kinect_2022_0801_4_cat_ring24000_and_real_trim0.04/demo_0828_000/-1_icp_smooth"
    # frames = [
    #     *list(range(19, 38)),
    #     *list(range(92, 106)),
    #     *list(range(116, 118)),
    #     *list(range(138, 145)),
    #     *list(range(170, 180))
    #     *list(range(218, 248)),
    #     *list(range(266, 289)),
    # ]
    img_files = sorted(glob.glob(osp.join(result_dir, "*png")))
    imgs = []
    for frame in frames:
        imgs.append(imageio.imread(img_files[frame]))
    imageio.mimwrite(osp.join(result_dir, "demo_selected.mp4"), imgs)


if __name__ == '__main__':
    main()
