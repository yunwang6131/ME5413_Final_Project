#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import Counter, defaultdict

import rospy
import cv2
import pytesseract
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Bool, Int32MultiArray

import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs

from sklearn.cluster import DBSCAN


MIN_OCR_CONF = 70


def _ocr_digit_psm10_conf(roi_bin):
    cfg = r"--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
    try:
        data = pytesseract.image_to_data(
            roi_bin, config=cfg, output_type=pytesseract.Output.DICT #Tesseract 对图像做 OCR;output_type=DICT 表示返回 Python 字典；image_to_data 会返回更详细的信息，不只是文字，还包括每个识别块的置信度
        )
    except pytesseract.TesseractError: #如果 OCR 调用失败，直接返回
        return None, -1

    best_d, best_c = None, -1 #初始化最佳数字和置信度为 None 和 -1
    for i in range(len(data.get("text", []))): #遍历 OCR 结果中的每个字符
        raw = (data["text"][i] or "").strip() #取出第 i 个识别文本，如果为空就变成空字符串，strip()去掉字符串开头和结尾
        if len(raw) != 1 or not raw.isdigit(): #如果文本长度不为 1 或者不是数字，就跳过
            continue
        try:
            c = int(float(data["conf"][i])) #将置信度转换为整数
        except (ValueError, IndexError, TypeError):
            continue
        if c > best_c: #如果当前置信度大于最佳置信度，就更新最佳置信度和最佳数字
            best_c, best_d = c, raw
    return best_d, best_c #返回最佳数字和最佳置信度 


def _bbox_area(r):
    """计算矩形框的面积，输入矩形 r=(x,y,w,h)，返回 w*h"""
    return r[2] * r[3] 


def _fully_covers(outer, inner):
    """判断外框是否完全覆盖内框,返回的是true和false"""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        ox <= ix and
        oy <= iy and
        ox + ow >= ix + iw and
        oy + oh >= iy + ih
    )


def _remove_larger_when_fully_covers(candidates):
    """输入一批候选数字框，如果一个大框完全包住一个更小的框，就删掉大框，保留更紧致的小框。"""
    n = len(candidates)
    if n <= 1:
        return candidates #如果候选框数量小于等于1，直接返回

    remove = set() #记录要删除的候选框下标
    for i in range(n): #双重循环，两两比较所有框
        for j in range(n): 
            if i == j or i in remove or j in remove:
                continue

            ri = (
                candidates[i]["x"],
                candidates[i]["y"],
                candidates[i]["w"],
                candidates[i]["h"],
            )
            rj = (
                candidates[j]["x"],
                candidates[j]["y"],
                candidates[j]["w"],
                candidates[j]["h"],
            )

            ai = _bbox_area(ri)
            aj = _bbox_area(rj)

            if _fully_covers(ri, rj) and ai > aj:
                remove.add(i)
            elif _fully_covers(rj, ri) and aj > ai:
                remove.add(j)

    return [candidates[k] for k in range(n) if k not in remove]


class BoxMapperFusion:
    def __init__(self):
        rospy.init_node("box_mapper")

        # =========================
        # Topics / Frames
        # =========================
        self.image_topic = rospy.get_param("~image_topic", "/front/image_raw")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/mid/points")
        self.map_topic = rospy.get_param("~map_topic", "/map")

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.camera_frame = rospy.get_param("~camera_frame", "front_camera_optical")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # =========================
        # Camera intrinsics
        # =========================
        self.fx = rospy.get_param("~fx", 554.254691191187)
        self.fy = rospy.get_param("~fy", 554.254691191187)
        self.cx = rospy.get_param("~cx", 320.5)
        self.cy = rospy.get_param("~cy", 256.5)

        # =========================
        # Vision params
        # =========================
        self.ground_z = rospy.get_param("~ground_z", 0.0)
        self.process_every_n = rospy.get_param("~process_every_n", 5)
        if self.process_every_n < 1:
            self.process_every_n = 1

        # 点云每N帧处理一次
        self.lidar_process_every_n = rospy.get_param("~lidar_process_every_n", 5)
        if self.lidar_process_every_n < 1:
            self.lidar_process_every_n = 1

        # =========================
        # LiDAR params
        # =========================
        self.cluster_eps = rospy.get_param("~cluster_eps", 0.5)
        self.min_samples = rospy.get_param("~min_samples", 30)
        self.z_threshold = rospy.get_param("~z_threshold", 0.1)
        self.map_tolerance = rospy.get_param("~map_tolerance", 1)

        # 最小改动：新增两个参数，只用于聚类后按离墙距离过滤整簇
        self.wall_clearance = rospy.get_param("~wall_clearance", 0.45)
        self.occ_threshold = rospy.get_param("~occ_threshold", 50)

        # =========================
        # Fusion / tracking params
        # =========================
        self.lidar_track_merge_dist = rospy.get_param("~lidar_track_merge_dist", 1.0)
        self.lidar_assoc_dist = rospy.get_param("~lidar_assoc_dist", 1.0)

        self.min_confirm_frames = rospy.get_param("~min_confirm_frames", 2)
        self.max_miss_frames = rospy.get_param("~max_miss_frames", 15)

        self.min_digit_vote_weight = rospy.get_param("~min_digit_vote_weight", 2.0)
        self.min_digit_margin = rospy.get_param("~min_digit_margin", 0.8)
        self.max_visual_assign_dist = rospy.get_param("~max_visual_assign_dist", 12.0)
        self.min_consecutive_digit_hits = rospy.get_param("~min_consecutive_digit_hits", 4)

        # 箱子实际尺寸（边长，单位 m）
        self.box_size = rospy.get_param("~box_size", 0.8)

        # 重叠判定裕量
        self.box_overlap_margin = rospy.get_param("~box_overlap_margin", 0.05)

        # 默认关闭opencv调试窗口，减少卡顿
        self.show_debug_image = rospy.get_param("~show_debug_image", False)

        # marker 默认是否显示
        self.marker_enabled = rospy.get_param("~marker_enabled", True)

        # =========================
        # Runtime data
        # =========================
        self.bridge = CvBridge()
        self.frame_count = 0
        self.lidar_frame_count = 0

        self.static_map = None
        self.map_info = None

        self.next_box_id = 1
        self.lidar_boxes = []

        self.last_least_digit_msg = ""
        self.shutdown_requested = False

        # =========================
        # TF
        # =========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # =========================
        # ROS pub/sub
        # =========================
        self.map_sub = rospy.Subscriber(
            self.map_topic, OccupancyGrid, self.map_callback, queue_size=1
        )
        self.lidar_sub = rospy.Subscriber(
            self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )

        self.marker_pub = rospy.Publisher(
            "/box_markers", MarkerArray, queue_size=1, latch=True
        )

        self.least_digit_pub = rospy.Publisher(
            "/least_frequent_digit", String, queue_size=1, latch=True
        )

        self.box_count_pub = rospy.Publisher(
            "/mission/box_counts", Int32MultiArray, queue_size=1, latch=True
        )

        self.shutdown_sub = rospy.Subscriber(
            "/box_mapper_shutdown", Bool, self.shutdown_callback, queue_size=1
        )

        self.marker_enable_sub = rospy.Subscriber(
            "/box_mapper_marker_enable", Bool, self.marker_enable_callback, queue_size=1
        )

        rospy.loginfo("box_mapper started")

    # =========================================================
    # External shutdown
    # =========================================================
    def shutdown_callback(self, msg):
        if msg.data and not self.shutdown_requested:
            self.shutdown_requested = True
            rospy.signal_shutdown("Shutdown requested from /box_mapper_shutdown")

    # =========================================================
    # Marker switch
    # =========================================================
    def marker_enable_callback(self, msg):
        self.marker_enabled = bool(msg.data)

        if self.marker_enabled:
            self.publish_markers()
        else:
            self.clear_markers()

    def clear_markers(self):
        arr = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)
        self.marker_pub.publish(arr)

    # =========================================================
    # Map
    # =========================================================
    def map_callback(self, msg):
        self.static_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

    # =========================================================
    # LiDAR main callback
    # =========================================================
    def lidar_callback(self, cloud_msg):
        if self.shutdown_requested:
            return

        if not self._should_process_lidar():
            return

        if self.static_map is None or self.map_info is None:
            return

        cloud_out = self._transform_cloud_to_map(cloud_msg)
        if cloud_out is None:
            return

        pts = self._extract_valid_lidar_points(cloud_out)

        if len(pts) < self.min_samples:
            self.age_tracks_without_measurement()
            self._publish_outputs()
            return

        centroids = self._cluster_points_to_centroids(pts)
        self.update_lidar_tracks(centroids)
        self._publish_outputs()

    def publish_box_counts(self):
        confirmed_boxes = [b for b in self.lidar_boxes if b["status"] == "confirmed"]
        stable_digits = [b["best_digit"] for b in confirmed_boxes if b["best_digit"] in ["1", "2", "3", "4"]]

        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for d in stable_digits:
            counts[int(d)] += 1

        msg = Int32MultiArray()
        msg.data = [counts[1], counts[2], counts[3], counts[4]]
        self.box_count_pub.publish(msg)

    def _should_process_lidar(self):
        self.lidar_frame_count += 1
        return (self.lidar_frame_count % self.lidar_process_every_n) == 0

    def _transform_cloud_to_map(self, cloud_msg):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                rospy.Duration(0.1)
            )
            return tf2_sensor_msgs.do_transform_cloud(cloud_msg, transform)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "LiDAR TF failed: %s", str(e))
            return None

    def _extract_valid_lidar_points(self, cloud_out):
        pts = []
        t = self.map_tolerance

        for p in pc2.read_points(
            cloud_out, field_names=("x", "y", "z"), skip_nans=True
        ):
            gx, gy, gz = p[0], p[1], p[2]

            if gz < self.z_threshold:
                continue

            mx = int((gx - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((gy - self.map_info.origin.position.y) / self.map_info.resolution)

            if (t <= mx < self.map_info.width - t) and (t <= my < self.map_info.height - t):
                local_region = self.static_map[my - t: my + t + 1, mx - t: mx + t + 1]
                if np.all(local_region == 0):
                    pts.append([gx, gy])

        return pts

    def _cluster_points_to_centroids(self, pts):
        X = np.array(pts)
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples).fit(X)
        labels = db.labels_

        centroids = []
        for label in set(labels):
            if label == -1:
                continue

            cluster_pts = X[labels == label]
            centroid = np.mean(cluster_pts, axis=0)

            # 最小改动：聚类后，如果整簇中心离墙太近，则丢弃该簇
            mx = int((centroid[0] - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((centroid[1] - self.map_info.origin.position.y) / self.map_info.resolution)

            if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
                continue

            clearance_cells = int(self.wall_clearance / self.map_info.resolution)
            x0 = max(0, mx - clearance_cells)
            x1 = min(self.map_info.width, mx + clearance_cells + 1)
            y0 = max(0, my - clearance_cells)
            y1 = min(self.map_info.height, my + clearance_cells + 1)

            local_region = self.static_map[y0:y1, x0:x1]

            if np.any(local_region >= self.occ_threshold):
                continue

            centroids.append((centroid[0], centroid[1]))

        return centroids

    def _publish_outputs(self):
        self.publish_markers()
        self.publish_least_frequent_digit()
        self.publish_box_counts()

    def age_tracks_without_measurement(self):
        for box in self.lidar_boxes:
            box["miss_count"] += 1
        self.remove_dead_tracks()
        self.merge_overlapping_boxes()

    def update_lidar_tracks(self, centroids):
        for box in self.lidar_boxes:
            box["miss_count"] += 1

        used_track_ids = set()

        for cx, cy in centroids:
            best_box = None
            best_d = 1e9

            for box in self.lidar_boxes:
                if box["id"] in used_track_ids:
                    continue
                d = math.hypot(cx - box["x"], cy - box["y"])
                if d < best_d and d < self.lidar_track_merge_dist:
                    best_d = d
                    best_box = box

            if best_box is not None:
                best_box["x"] = 0.75 * best_box["x"] + 0.25 * cx
                best_box["y"] = 0.75 * best_box["y"] + 0.25 * cy
                best_box["seen_count"] += 1
                best_box["miss_count"] = 0

                if best_box["seen_count"] >= self.min_confirm_frames:
                    best_box["status"] = "confirmed"

                used_track_ids.add(best_box["id"])
            else:
                self.lidar_boxes.append({
                    "id": self.next_box_id,
                    "x": cx,
                    "y": cy,
                    "seen_count": 1,
                    "miss_count": 0,
                    "status": "tentative",
                    "digit_scores": defaultdict(float),
                    "digit_counts": defaultdict(int),
                    "best_digit": "?",
                    "best_digit_score": 0.0,
                    "best_conf": -1,
                    "last_dist": -1.0,
                    "last_seen_digit": None,
                    "consecutive_digit_hits": 0
                })
                self.next_box_id += 1

        self.remove_dead_tracks()
        self.merge_overlapping_boxes()

    def remove_dead_tracks(self):
        kept = []
        for b in self.lidar_boxes:
            if b["status"] == "confirmed":
                if "last_seen_digit" not in b:
                    b["last_seen_digit"] = None
                if "consecutive_digit_hits" not in b:
                    b["consecutive_digit_hits"] = 0
                kept.append(b)
            else:
                if b["miss_count"] <= self.max_miss_frames:
                    if "last_seen_digit" not in b:
                        b["last_seen_digit"] = None
                    if "consecutive_digit_hits" not in b:
                        b["consecutive_digit_hits"] = 0
                    kept.append(b)
        self.lidar_boxes = kept

    # =========================================================
    # Box overlap / merge
    # =========================================================
    def boxes_overlap(self, a, b):
        thresh = self.box_size + self.box_overlap_margin
        return (
            abs(a["x"] - b["x"]) < thresh and
            abs(a["y"] - b["y"]) < thresh
        )

    def merge_two_boxes(self, a, b):
        wa = max(1, a["seen_count"])
        wb = max(1, b["seen_count"])
        wsum = wa + wb

        a["x"] = (a["x"] * wa + b["x"] * wb) / float(wsum)
        a["y"] = (a["y"] * wa + b["y"] * wb) / float(wsum)

        a["seen_count"] = max(a["seen_count"], b["seen_count"])
        a["miss_count"] = min(a["miss_count"], b["miss_count"])

        if a["status"] == "confirmed" or b["status"] == "confirmed":
            a["status"] = "confirmed"
        else:
            a["status"] = "tentative"

        for k, v in b["digit_scores"].items():
            a["digit_scores"][k] += v

        for k, v in b["digit_counts"].items():
            a["digit_counts"][k] += v

        a["best_conf"] = max(a["best_conf"], b["best_conf"])

        if a["last_dist"] < 0:
            a["last_dist"] = b["last_dist"]
        elif b["last_dist"] >= 0:
            a["last_dist"] = min(a["last_dist"], b["last_dist"])

        a_last_digit = a.get("last_seen_digit")
        b_last_digit = b.get("last_seen_digit")
        a_hits = a.get("consecutive_digit_hits", 0)
        b_hits = b.get("consecutive_digit_hits", 0)

        if a_last_digit == b_last_digit:
            a["last_seen_digit"] = a_last_digit
            a["consecutive_digit_hits"] = max(a_hits, b_hits)
        elif a_hits >= b_hits:
            a["last_seen_digit"] = a_last_digit
            a["consecutive_digit_hits"] = a_hits
        else:
            a["last_seen_digit"] = b_last_digit
            a["consecutive_digit_hits"] = b_hits

        self.update_box_best_digit(a)

    def merge_overlapping_boxes(self):
        changed = True
        while changed:
            changed = False
            n = len(self.lidar_boxes)

            for i in range(n):
                if i >= len(self.lidar_boxes):
                    break

                a = self.lidar_boxes[i]
                merged = False

                for j in range(i + 1, len(self.lidar_boxes)):
                    b = self.lidar_boxes[j]

                    if self.boxes_overlap(a, b):
                        keep_a = (
                            (a["status"] == "confirmed" and b["status"] != "confirmed") or
                            (a["status"] == b["status"] and a["seen_count"] >= b["seen_count"])
                        )

                        if keep_a:
                            self.merge_two_boxes(a, b)
                            del self.lidar_boxes[j]
                        else:
                            self.merge_two_boxes(b, a)
                            del self.lidar_boxes[i]

                        changed = True
                        merged = True
                        break

                if merged:
                    break

    # =========================================================
    # Vision main callback
    # =========================================================
    def image_callback(self, msg):
        if self.shutdown_requested:
            return

        if not self._should_process_image():
            return

        frame = self._read_ros_image(msg)
        if frame is None:
            return

        detections, vis_frame = self.detect_digits_and_redpoints(frame)
        self._fuse_visual_detections(detections, msg.header.stamp)

        self._publish_outputs()

        if self.show_debug_image:
            cv2.imshow("ROS Detection + Mapping Fusion", vis_frame)
            cv2.waitKey(1)

    def _should_process_image(self):
        self.frame_count += 1
        return (self.frame_count % self.process_every_n) == 0

    def _read_ros_image(self, msg):
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", str(e))
            return None

    def _fuse_visual_detections(self, detections, stamp):
        for det in detections:
            ground_pt = self.project_pixel_to_ground(
                det["dot_x"], det["dot_y"], stamp
            )
            if ground_pt is None:
                continue

            gx, gy, dist = ground_pt

            if dist > self.max_visual_assign_dist:
                continue

            self.assign_digit_to_lidar_box(gx, gy, det["digit"], det["conf"], dist)

    def detect_digits_and_redpoints(self, frame):
        img_h, img_w = frame.shape[:2]
        total_pixels = img_h * img_w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 10
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_digit_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_percent = (area / total_pixels) * 100.0

            if 0.05 < area_percent < 15.0:
                x, y, w, h = cv2.boundingRect(cnt)

                dw, dh = 3, 3
                ex_x = max(0, x - dw)
                ex_y = max(0, y - dh)
                ex_w = min(img_w - ex_x, w + 2 * dw)
                ex_h = min(img_h - ex_y, h + 2 * dh)

                aspect_ratio = float(ex_w) / float(ex_h)
                if 0.1 < aspect_ratio < 1.5:
                    roi = gray[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]
                    _, roi_bin = cv2.threshold(
                        roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    digit, ocr_conf = _ocr_digit_psm10_conf(roi_bin)

                    if digit is not None and ocr_conf >= MIN_OCR_CONF:
                        raw_digit_regions.append({
                            "x": ex_x,
                            "y": ex_y,
                            "w": ex_w,
                            "h": ex_h,
                            "digit": digit,
                            "conf": ocr_conf
                        })

        final_regions = _remove_larger_when_fully_covers(raw_digit_regions)

        detections = []
        vis_frame = frame.copy()

        for r in final_regions:
            ex_x, ex_y, ex_w, ex_h = r["x"], r["y"], r["w"], r["h"]
            digit, ocr_conf = r["digit"], r["conf"]

            draw_x = ex_x + 3
            draw_y = ex_y + 3
            draw_w = max(1, ex_w - 6)
            draw_h = max(1, ex_h - 6)

            dot_x = int(draw_x + draw_w / 2)
            dot_y = int(draw_y + 1.5 * draw_h)

            dot_x = min(max(0, dot_x), img_w - 1)
            dot_y = min(max(0, dot_y), img_h - 1)

            detections.append({
                "digit": digit,
                "conf": ocr_conf,
                "dot_x": dot_x,
                "dot_y": dot_y
            })

            if self.show_debug_image:
                cv2.rectangle(
                    vis_frame,
                    (draw_x, draw_y),
                    (draw_x + draw_w, draw_y + draw_h),
                    (0, 255, 0), 2
                )
                cv2.circle(vis_frame, (dot_x, dot_y), 5, (0, 0, 255), -1)

                label = "N:{}({})".format(digit, ocr_conf)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                ty = max(draw_y - 6, th + 4)

                cv2.rectangle(
                    vis_frame,
                    (draw_x, ty - th - 4),
                    (draw_x + tw + 4, ty + 2),
                    (255, 255, 255), -1
                )
                cv2.putText(
                    vis_frame,
                    label,
                    (draw_x + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 200),
                    2
                )

        return detections, vis_frame

    def project_pixel_to_ground(self, u, v, stamp):
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0

        p0_cam = PointStamped()
        p0_cam.header.stamp = stamp
        p0_cam.header.frame_id = self.camera_frame
        p0_cam.point.x = 0.0
        p0_cam.point.y = 0.0
        p0_cam.point.z = 0.0

        p1_cam = PointStamped()
        p1_cam.header.stamp = stamp
        p1_cam.header.frame_id = self.camera_frame
        p1_cam.point.x = ray_x
        p1_cam.point.y = ray_y
        p1_cam.point.z = ray_z

        try:
            p0_map = self.tf_buffer.transform(
                p0_cam, self.map_frame, rospy.Duration(0.2)
            )
            p1_map = self.tf_buffer.transform(
                p1_cam, self.map_frame, rospy.Duration(0.2)
            )
        except Exception as e:
            rospy.logwarn_throttle(1.0, "pixel->ground TF failed: %s", str(e))
            return None

        x0, y0, z0 = p0_map.point.x, p0_map.point.y, p0_map.point.z
        x1, y1, z1 = p1_map.point.x, p1_map.point.y, p1_map.point.z

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        if abs(dz) < 1e-6:
            return None

        t = (self.ground_z - z0) / dz
        if t <= 0:
            return None

        gx = x0 + t * dx
        gy = y0 + t * dy
        horizontal_dist = math.hypot(gx - x0, gy - y0)

        return gx, gy, horizontal_dist

    # =========================================================
    # Fusion logic
    # =========================================================
    def assign_digit_to_lidar_box(self, x, y, digit, conf, dist):
        confirmed_boxes = [
            b for b in self.lidar_boxes if b["status"] == "confirmed"
        ]
        if len(confirmed_boxes) == 0:
            return

        best_box = None
        best_d = 1e9

        for box in confirmed_boxes:
            d = math.hypot(x - box["x"], y - box["y"])
            if d < best_d and d < self.lidar_assoc_dist:
                best_d = d
                best_box = box

        if best_box is None:
            return

        if best_box.get("last_seen_digit") == digit:
            best_box["consecutive_digit_hits"] += 1
        else:
            best_box["last_seen_digit"] = digit
            best_box["consecutive_digit_hits"] = 1

        if best_box["consecutive_digit_hits"] < self.min_consecutive_digit_hits:
            return

        weight = max(0.0, min(1.0, conf / 100.0))
        distance_factor = 1.0 / (1.0 + 0.08 * max(0.0, dist - 2.0))
        final_weight = weight * distance_factor

        best_box["digit_scores"][digit] += final_weight
        best_box["digit_counts"][digit] += 1
        best_box["last_dist"] = dist

        if conf > best_box["best_conf"]:
            best_box["best_conf"] = conf

        self.update_box_best_digit(best_box)

    def update_box_best_digit(self, box):
        if len(box["digit_scores"]) == 0:
            box["best_digit"] = "?"
            box["best_digit_score"] = 0.0
            return

        sorted_items = sorted(
            box["digit_scores"].items(),
            key=lambda kv: kv[1],
            reverse=True
        )

        top_digit, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = top_score - second_score

        if top_score < self.min_digit_vote_weight:
            box["best_digit"] = "?"
            box["best_digit_score"] = top_score
            return

        if margin < self.min_digit_margin:
            box["best_digit"] = "?"
            box["best_digit_score"] = top_score
            return

        box["best_digit"] = top_digit
        box["best_digit_score"] = top_score

    # =========================================================
    # Least frequent digit publisher
    # =========================================================
    def publish_least_frequent_digit(self):
        confirmed_boxes = [b for b in self.lidar_boxes if b["status"] == "confirmed"]
        stable_digits = [b["best_digit"] for b in confirmed_boxes if b["best_digit"] != "?"]

        if len(stable_digits) == 0:
            return

        counter = Counter(stable_digits)
        min_count = min(counter.values())
        least_common_digits = sorted([k for k, v in counter.items() if v == min_count])

        msg = String()
        if len(least_common_digits) == 1:
            msg.data = least_common_digits[0]
        else:
            msg.data = ",".join(least_common_digits)

        self.least_digit_pub.publish(msg)

        if msg.data != self.last_least_digit_msg:
            self.last_least_digit_msg = msg.data

    # =========================================================
    # Visualization
    # =========================================================
    def publish_markers(self):
        if not self.marker_enabled:
            return

        arr = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        idx = 0

        try:
            tf_base = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.2)
            )

            car = Marker()
            car.header.frame_id = self.map_frame
            car.header.stamp = rospy.Time.now()
            car.ns = "robot"
            car.id = idx
            car.type = Marker.SPHERE
            car.action = Marker.ADD
            car.pose.position.x = tf_base.transform.translation.x
            car.pose.position.y = tf_base.transform.translation.y
            car.pose.position.z = 0.2
            car.pose.orientation.w = 1.0
            car.scale.x = 0.4
            car.scale.y = 0.4
            car.scale.z = 0.4
            car.color.a = 1.0
            car.color.r = 0.0
            car.color.g = 1.0
            car.color.b = 0.0
            arr.markers.append(car)
            idx += 1

            car_text = Marker()
            car_text.header.frame_id = self.map_frame
            car_text.header.stamp = rospy.Time.now()
            car_text.ns = "robot_label"
            car_text.id = idx
            car_text.type = Marker.TEXT_VIEW_FACING
            car_text.action = Marker.ADD
            car_text.pose.position.x = tf_base.transform.translation.x
            car_text.pose.position.y = tf_base.transform.translation.y
            car_text.pose.position.z = 0.8
            car_text.pose.orientation.w = 1.0
            car_text.scale.z = 0.35
            car_text.color.a = 1.0
            car_text.color.r = 0.0
            car_text.color.g = 1.0
            car_text.color.b = 0.0
            car_text.text = "Robot"
            arr.markers.append(car_text)
            idx += 1

        except Exception as e:
            rospy.logwarn_throttle(1.0, "robot TF lookup failed: %s", str(e))

        confirmed_boxes = [b for b in self.lidar_boxes if b["status"] == "confirmed"]

        for box in confirmed_boxes:
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = rospy.Time.now()
            m.ns = "boxes"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = box["x"]
            m.pose.position.y = box["y"]
            m.pose.position.z = self.box_size / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = self.box_size
            m.scale.y = self.box_size
            m.scale.z = self.box_size
            m.color.a = 0.85

            if box["best_digit"] == "?":
                if box["miss_count"] == 0:
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                else:
                    m.color.r = 0.6
                    m.color.g = 0.6
                    m.color.b = 0.0
            else:
                if box["miss_count"] == 0:
                    m.color.r = 1.0
                    m.color.g = 0.0
                    m.color.b = 0.0
                else:
                    m.color.r = 0.5
                    m.color.g = 0.0
                    m.color.b = 0.0

            arr.markers.append(m)
            idx += 1

            t = Marker()
            t.header.frame_id = self.map_frame
            t.header.stamp = rospy.Time.now()
            t.ns = "box_labels"
            t.id = idx
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = box["x"]
            t.pose.position.y = box["y"]
            t.pose.position.z = self.box_size + 0.45
            t.pose.orientation.w = 1.0
            t.scale.z = 0.36
            t.color.a = 1.0
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0

            score_text = ",".join(
                ["{}:{:.1f}".format(k, v) for k, v in sorted(box["digit_scores"].items())]
            )
            if score_text == "":
                score_text = "None"

            if box["miss_count"] == 0:
                state_text = "live"
            else:
                state_text = "kept"

            t.text = "Box {} | {} | Num {} | score[{}] | hit {}".format(
                box["id"],
                state_text,
                box["best_digit"],
                score_text,
                box.get("consecutive_digit_hits", 0)
            )
            arr.markers.append(t)
            idx += 1

        stable_digits = [b["best_digit"] for b in confirmed_boxes if b["best_digit"] != "?"]

        if len(stable_digits) > 0:
            counter = Counter(stable_digits)
            min_count = min(counter.values())
            least_common_digits = sorted([k for k, v in counter.items() if v == min_count])

            summary = Marker()
            summary.header.frame_id = self.map_frame
            summary.header.stamp = rospy.Time.now()
            summary.ns = "summary_label"
            summary.id = idx
            summary.type = Marker.TEXT_VIEW_FACING
            summary.action = Marker.ADD
            summary.pose.orientation.w = 1.0
            summary.scale.z = 0.5
            summary.color.a = 1.0
            summary.color.r = 0.0
            summary.color.g = 1.0
            summary.color.b = 1.0

            try:
                tf_base = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.2)
                )
                summary.pose.position.x = tf_base.transform.translation.x
                summary.pose.position.y = tf_base.transform.translation.y + 1.2
                summary.pose.position.z = 1.5
            except Exception:
                summary.pose.position.x = 0.0
                summary.pose.position.y = 0.0
                summary.pose.position.z = 1.5

            stats_text = "  ".join(
                ["{}:{}".format(k, v) for k, v in sorted(counter.items())]
            )

            if len(least_common_digits) == 1:
                summary.text = "Least frequent: {} ({}) | {}".format(
                    least_common_digits[0], min_count, stats_text
                )
            else:
                summary.text = "Least frequent: {} ({}) | {}".format(
                    ",".join(least_common_digits), min_count, stats_text
                )

            arr.markers.append(summary)
            idx += 1

        self.marker_pub.publish(arr)


if __name__ == "__main__":
    try:
        BoxMapperFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()