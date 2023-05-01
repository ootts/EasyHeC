import imageio
import subprocess

import numpy as np
import cv2


class PointDrawer(object):
    def __init__(self, window_name="Point Drawer", screen_scale=1.0, sam_checkpoint=""):
        self.window_name = window_name  # Name for our window
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = np.empty((0, 2))  # List of points defining our polygon
        self.labels = np.empty([0], dtype=int)
        self.screen_scale = screen_scale * 1.2

        ################  ↓ Step : build SAM  ↓  ##############
        device = "cuda"
        model_type = "default"

        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.mask = None

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                label = 0
            else:
                label = 1
            print(f"Adding point #{len(self.points)} with position({x},{y}), label {label}")
            self.points = np.concatenate((self.points, np.array([[x, y]])), axis=0)
            self.labels = np.concatenate((self.labels, np.array([label])), axis=0)

            self.detect()

    def detect(self):
        input_point = self.points / self.ratio
        input_label = self.labels.astype(int)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        maxidx = np.argmax(scores)
        mask = masks[maxidx]
        self.mask = mask.copy()

    def run(self, rgb):
        self.predictor.set_image(rgb)

        image_to_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        output = subprocess.check_output(["xrandr"]).decode("utf-8")
        current_mode = [line for line in output.splitlines() if "*" in line][0]
        screen_width, screen_height = [int(x) for x in current_mode.split()[0].split("x")]
        scale = self.screen_scale
        screen_w = int(screen_width / scale)
        screen_h = int(screen_height / scale)

        image_h, image_w = image_to_show.shape[:2]
        ratio = min(screen_w / image_w, screen_h / image_h)
        self.ratio = ratio
        target_size = (int(image_w * ratio), int(image_h * ratio))
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
            if self.points.shape[0] > 0:
                for ptidx, pt in enumerate(self.points):
                    color = (0, 255, 0) if self.labels[ptidx] == 1 else (0, 0, 255)
                    tmp = cv2.circle(tmp, (int(pt[0]), int(pt[1])), radius=5,
                                     color=color,
                                     thickness=-1)
            if self.mask is not None:
                mask_to_show = cv2.resize(self.mask.astype(np.uint8), target_size).astype(bool)
                tmp = tmp / 255.0
                tmp[mask_to_show] *= 0.5
                tmp[mask_to_show] += 0.5
                tmp = (tmp * 255).astype(np.uint8)
            cv2.imshow(self.window_name, tmp)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        return self.points, self.labels, self.mask


def main():
    img_path = "data/xarm7/example/color/000000.png"
    img_path = "/home/linghao/PycharmProjects/cam_robot_calib/data/realsense/20230121_225925/rgb_000001.png"
    img_path = "/home/linghao/Pictures/dog.jpeg"
    rgb = imageio.imread_v2(img_path)
    pointdrawer = PointDrawer(sam_checkpoint="models/sam/sam_vit_h_4b8939.pth")
    points, labels, mask = pointdrawer.run(rgb)
    print()


if __name__ == '__main__':
    main()
