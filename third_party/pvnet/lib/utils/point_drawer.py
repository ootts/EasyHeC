import numpy as np
import cv2


# ============================================================================

# CANVAS_SIZE = (600, 800)

# FINAL_LINE_COLOR = (255, 255, 255)
# WORKING_LINE_COLOR = (127, 127, 127)


# ============================================================================

class PointDrawer(object):
    def __init__(self, window_name, canvas_size):
        """
        canvas_size: h,w
        """
        self.window_name = window_name  # Name for our window
        self.canvas_size = canvas_size
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = np.empty((0, 2))  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points = np.concatenate((self.points, np.array([[x, y]])), axis=0)
        # elif event == cv2.EVENT_LBUTTONDBLCLK:
        #     self.points = self.points[:-1, :]
        #     Right click means we're done
        # print("Completing polygon with %d points." % len(self.points))
        # self.done = True

    def run(self, image_to_show):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image_to_show)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while (not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            tmp = image_to_show.copy()
            tmp = cv2.circle(tmp, self.current, radius=2,
                             color=(0, 0, 255),
                             thickness=-1)
            if self.points.shape[0] > 0:
                # Draw all the current polygon segments
                # cv2.polylines(image_to_show, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                for ptidx, pt in enumerate(self.points):
                    tmp = cv2.circle(tmp, (int(pt[0]), int(pt[1])), radius=2,
                                     color=(0, 0, 255),
                                     thickness=-1)
                    tmp = cv2.putText(tmp, str(ptidx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                # And  also show what the current segment would look like
                # cv2.line(image_to_show, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, tmp)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        # canvas = np.zeros(self.canvas_size, np.uint8)
        # of a filled polygon
        # if (len(self.points) > 0):
        #     cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        # cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        # cv2.waitKey()

        # cv2.destroyWindow(self.window_name)
        return self.points


# ============================================================================

if __name__ == "__main__":
    pd = PointDrawer("Polygon", (720, 1280))
    pts = pd.run(np.zeros((720, 1280, 3)))
    # cv2.imwrite("polygon.png", image)
    print("pts = %s" % pts)
