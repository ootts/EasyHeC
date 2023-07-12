import cv2

import loguru
import numpy as np
import pyrealsense2 as rs


class RealSenseAPI:
    pipeline, profile, align = None, None, None

    @staticmethod
    def setup_realsense():
        loguru.logger.info("Setting up RealSense")
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
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

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)
        profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
        align_to = rs.stream.color
        align = rs.align(align_to)
        for _ in range(10):  # wait for white balance to stabilize
            frames = pipeline.wait_for_frames()
        return pipeline, profile, align

    @staticmethod
    def capture_data():
        if RealSenseAPI.pipeline is None or RealSenseAPI.profile is None or RealSenseAPI.align is None:
            RealSenseAPI.pipeline, RealSenseAPI.profile, RealSenseAPI.align = RealSenseAPI.setup_realsense()

        pipeline, profile, align = RealSenseAPI.pipeline, RealSenseAPI.profile, RealSenseAPI.align
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]])
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            break
        return rgb, K
