#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError


class RedColorDetector:
    def __init__(self):
        rospy.init_node("red_color_detector_node", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/image_raw", Image, self.callback)
        self.pub_red_detected = rospy.Publisher(
            "/red_detected", Bool, queue_size=1, latch=True
        )
        self.pub_red_left = rospy.Publisher(
            "/red_left", Bool, queue_size=1, latch=False
        )
        self.last_state = None

        # 红色在 HSV 中跨越 0 度附近，通常要用两段阈值
        self.lower_red_1 = np.array([0, 80, 60], dtype=np.uint8)
        self.upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        self.lower_red_2 = np.array([170, 80, 60], dtype=np.uint8)
        self.upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        # 至少需要占总像素这么多比例才认为“有红色”，防止噪声误检
        self.min_red_ratio = 0.005  # 0.5%

        rospy.loginfo("红色检测节点已启动；发布 /red_detected 和 /red_left (std_msgs/Bool)")

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
        mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), dtype=np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_pixels = int(cv2.countNonZero(red_mask))
        img_h, img_w = frame.shape[:2]
        total_pixels = img_h * img_w
        min_red_pixels = int(total_pixels * self.min_red_ratio)
        has_red = red_pixels >= min_red_pixels

        msg = Bool()
        msg.data = has_red
        self.pub_red_detected.publish(msg)

        # 仅在 1 -> 0 的瞬间发布“离开事件”
        if self.last_state is True and has_red is False:
            left_msg = Bool()
            left_msg.data = True
            self.pub_red_left.publish(left_msg)
            rospy.loginfo("红色物体离开画面: /red_left=True")

        if has_red != self.last_state:
            rospy.loginfo(f"红色检测状态: {has_red} (red_pixels={red_pixels})")
            self.last_state = has_red

        display = frame.copy()
        status_text = "RED DETECTED" if has_red else "NO RED"
        status_color = (0, 0, 255) if has_red else (0, 255, 0)
        cv2.putText(
            display,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
        )
        cv2.imshow("ROS Red Color Detection", display)
        cv2.waitKey(1)


if __name__ == "__main__":
    detector = RedColorDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("正在关闭红色检测节点...")
    finally:
        cv2.destroyAllWindows()