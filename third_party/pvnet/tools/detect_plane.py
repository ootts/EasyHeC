import numpy as np
import pyrealsense2 as rs
from easyhec.utils import utils_3d
from easyhec.utils.vis3d_ext import Vis3D

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
align_to = rs.stream.color
align = rs.align(align_to)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]])
for _ in range(20):
    pipeline.wait_for_frames()
vis3d = Vis3D(
    xyz_pattern=('x', '-y', '-z'),
    out_folder="dbg",
    sequence="detect_plane",
    # auto_increase=,
    # enable=,
)
while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) / 1000.0
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth_image)
    maxz = 1000
    plane_model, inliers = utils_3d.open3d_plane_segment_api(pts_rect[pts_rect[:, 2] < maxz], 0.02)
    vis3d.add_point_cloud(pts_rect[pts_rect[:, 2] < maxz][inliers], name='plane')
    vis3d.add_point_cloud(pts_rect[pts_rect[:, 2] < maxz],min_norm=0.0, name='all_points')
    # vis3d.add_point_cloud(pts_rect, name='all_points')
    print(np.array2string(plane_model, separator=","))
    vis3d.increase_scene_id()
    # break
