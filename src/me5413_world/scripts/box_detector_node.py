#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import pytesseract
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, Int32MultiArray, String

# Tesseract 置信度 0~100；低于此值不视为有效识别
MIN_OCR_CONF = 70


def _ocr_digit_psm10_conf(roi_bin):
    """--psm 10 单字，返回 (digit 或 None, confidence)。"""
    cfg = r"--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
    try:
        data = pytesseract.image_to_data(
            roi_bin, config=cfg, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        return None, -1

    best_d, best_c = None, -1
    for i in range(len(data.get("text", []))):
        raw = (data["text"][i] or "").strip()
        if len(raw) != 1 or not raw.isdigit():
            continue
        try:
            c = int(data["conf"][i])
        except (ValueError, IndexError):
            continue
        if c > best_c:
            best_c, best_d = c, raw
    return best_d, best_c


def _bbox_area(r):
    return r[2] * r[3]


def _fully_covers(outer, inner):
    """outer=(x,y,w,h) 是否完全覆盖 inner（边可重合）。"""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        ox <= ix
        and oy <= iy
        and ox + ow >= ix + iw
        and oy + oh >= iy + ih
    )


def _remove_larger_when_fully_covers(candidates):
    """
    若某框完全覆盖另一框，则去掉面积更大的那个。
    """
    n = len(candidates)
    if n <= 1:
        return candidates

    remove = set()
    for i in range(n):
        for j in range(n):
            if i == j or i in remove or j in remove:
                continue
            ri = (candidates[i]["x"], candidates[i]["y"], candidates[i]["w"], candidates[i]["h"])
            rj = (candidates[j]["x"], candidates[j]["y"], candidates[j]["w"], candidates[j]["h"])
            ai, aj = _bbox_area(ri), _bbox_area(rj)

            if _fully_covers(ri, rj) and ai > aj:
                remove.add(i)
            elif _fully_covers(rj, ri) and aj > ai:
                remove.add(j)

    return [candidates[k] for k in range(n) if k not in remove]


class BoxDetector:
    def __init__(self):
        rospy.init_node("box_detector_node", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/image_raw", Image, self.callback)
        self.pub_largest_digit = rospy.Publisher(
            "/digit_on_nearest_box", String, queue_size=1, latch=True
        )
        rospy.loginfo(
            "数字识别节点已启动；发布 /digit_on_nearest_box (std_msgs/String，面积最大绿框内数字)"
        )

    def callback(self, data):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        img_h, img_w = frame.shape[:2]
        total_pixels = img_h * img_w

        # 1. 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. 自适应二值化
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 51, 10)

        # 3. 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        raw_digit_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_percent = (area / total_pixels) * 100 

            # 过滤条件
            if 0.05 < area_percent < 15.0:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # OCR 外扩区域
                dw, dh = 3, 3
                ex_x = max(0, x - dw)
                ex_y = max(0, y - dh)
                ex_w = min(img_w - ex_x, w + 2 * dw)
                ex_h = min(img_h - ex_y, h + 2 * dh)

                aspect_ratio = float(ex_w) / ex_h
                
                if 0.1 < aspect_ratio < 1.5:
                    # ROI 识别
                    roi = gray[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
                    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    digit, ocr_conf = _ocr_digit_psm10_conf(roi_bin)

                    if digit is not None and ocr_conf >= MIN_OCR_CONF:
                        raw_digit_regions.append({
                            "x": ex_x, "y": ex_y, "w": ex_w, "h": ex_h,
                            "digit": digit, "conf": ocr_conf
                        })

        # 4. 后处理：去重
        final_regions = _remove_larger_when_fully_covers(raw_digit_regions)

        # 4.1 面积最大的绿色框（紧凑绘制框 draw_w*draw_h）对应的数字 → topic
        largest_digit = ""
        largest_area = -1.0
        for r in final_regions:
            ex_x, ex_y, ex_w, ex_h = r["x"], r["y"], r["w"], r["h"]
            draw_w = max(1, ex_w - 6)
            draw_h = max(1, ex_h - 6)
            a = float(draw_w * draw_h)
            if a > largest_area:
                largest_area = a
                largest_digit = r["digit"]
        msg_out = String()
        msg_out.data = largest_digit if largest_digit else ""
        self.pub_largest_digit.publish(msg_out)

        # 5. 绘制与日志输出
        for r in final_regions:
            ex_x, ex_y, ex_w, ex_h = r["x"], r["y"], r["w"], r["h"]
            digit, ocr_conf = r["digit"], r["conf"]

            # 还原紧凑坐标（去掉外扩）
            draw_x, draw_y = ex_x + 3, ex_y + 3
            draw_w, draw_h = max(1, ex_w - 6), max(1, ex_h - 6)

            # 计算红点坐标 (底边中点下方 0.5 倍高度处)
            dot_x = int(draw_x + draw_w / 2)
            dot_y = int(draw_y + 1.5 * draw_h)
            
            # 边界检查
            dot_x_clipped = min(max(0, dot_x), img_w - 1)
            dot_y_clipped = min(max(0, dot_y), img_h - 1)

            # A. 绘制识别框 (绿色)
            cv2.rectangle(frame, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), (0, 255, 0), 2)

            # B. 绘制红点
            cv2.circle(frame, (dot_x_clipped, dot_y_clipped), 5, (0, 0, 255), -1)

            # C. 绘制文字标签
            label = f"N:{digit}({ocr_conf})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(draw_y - 6, th + 4)
            cv2.rectangle(frame, (draw_x, ty - th - 4), (draw_x + tw + 4, ty + 2), (255, 255, 255), -1)
            cv2.putText(frame, label, (draw_x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)

        # 显示图像
        cv2.imshow("ROS Detection (Green: Digit, Red: Target Dot)", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    detector = BoxDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("正在关闭识别节点...")
    finally:
        cv2.destroyAllWindows()