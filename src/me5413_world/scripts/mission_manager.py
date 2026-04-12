#!/usr/bin/env python3
import math

import rospy
import std_srvs.srv
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32MultiArray, String
from actionlib_msgs.msg import GoalStatusArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion


def yaw_to_quat(yaw):
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    return qx, qy, qz, qw


class MissionManager:
    def __init__(self):
        self.state = "INIT"
        self.state_enter_time = rospy.Time.now()
        self.start_time = rospy.Time.now()

        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.45))
        self.yaw_tolerance = float(rospy.get_param("~yaw_tolerance", 0.2))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 60.0))
        self.max_retries = int(rospy.get_param("~max_retries", 2))
        self.tick_hz = float(rospy.get_param("~tick_hz", 2.0))
        self.init_wait = float(rospy.get_param("~init_wait", 2.0))
        self.unblock_wait = float(rospy.get_param("~unblock_wait", 1.0))
        self.stop_after_unblock = bool(rospy.get_param("~stop_after_unblock", False))
        self.lower_unlock_on_last_point = bool(
            rospy.get_param("~lower_unlock_on_last_point", True)
        )
        self.lower_last_point_tolerance = float(
            rospy.get_param("~lower_last_point_tolerance", self.goal_tolerance)
        )
        self.use_external_box_counts = bool(rospy.get_param("~use_external_box_counts", False))


        # Wait for AMCL convergence before starting any mission. Threshold = max
        # acceptable position covariance trace (m^2). 0.5 corresponds to ~0.7 m std dev.
        self.wait_for_amcl = bool(rospy.get_param("~wait_for_amcl", True))
        self.amcl_cov_threshold = float(rospy.get_param("~amcl_cov_threshold", 0.5))
        self.amcl_max_wait = float(rospy.get_param("~amcl_max_wait", 30.0))
        self.amcl_pos_cov_trace = None
        self.amcl_seen = False
        self._on_ramp = False
        self._last_amcl_pose = None
        self._jump_filter_active = False

        # list entries are [x, y, yaw]
        self.lower_waypoints = rospy.get_param(
            "~lower_waypoints",
            [
                [-20.0, -7.0, 0.0],
                [-15.0, -7.0, 0.0],
            ],
        )
        self.exit_goal = rospy.get_param("~exit_goal", [-12.0, -6.5, 0.0])
        self.ramp_waypoints = rospy.get_param(
            "~ramp_waypoints",
            [
                [-10.0, -5.0, 0.0],
                [-8.0, -2.0, 0.5],
            ],
        )
        # [box_id, x, y, yaw]
        self.door_observation_goals_raw = rospy.get_param(
            "~door_observation_goals",
            [
                [1, 7.5, -7.59, 3.14],
                [2, 7.5, -2.47, 3.14],
                [3, 7.5,  2.57, 3.14],
                [4, 7.5,  7.75, 3.14],
            ],
        )
        self.door_observation_goals = self._parse_final_goals(self.door_observation_goals_raw)

        self.door_entry_goals_raw = rospy.get_param(
            "~door_entry_goals",
            [
                [1, 9.0, -7.59, 3.14],
                [2, 9.0, -2.47, 3.14],
                [3, 9.0,  2.57, 3.14],
                [4, 9.0,  7.75, 3.14],
            ],
        )
        self.door_entry_goals = self._parse_final_goals(self.door_entry_goals_raw)

        # box id -> count
        self.box_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        default_counts = rospy.get_param("~box_counts_default", [2, 3, 1, 4])
        if len(default_counts) >= 4:
            self.box_counts = {
                1: int(default_counts[0]),
                2: int(default_counts[1]),
                3: int(default_counts[2]),
                4: int(default_counts[3]),
            }
        self._last_box_counts = dict(self.box_counts)

        self.current_pose = None
        self.move_base_status = None
        self.active_goal = None
        self.active_goal_name = ""
        self.goal_sent_time = None
        self.retry_count = 0

        self.route = []
        self.route_idx = -1
        self.unblock_sent = False

        self.target_digit = None
        self.target_door_id = None

        self.counting_enabled = True
        self.box_mapper_shutdown_sent = False

        self.door_ids_in_order = []
        self.current_door_search_idx = -1

        self.nearest_box_digit = ""
        self.last_nearest_box_digit_time = None
        self.verify_wait = float(rospy.get_param("~verify_wait", 2.0))

        self.target_digit = None
        self.latest_least_digit = None
        self.last_least_digit_time = None
        

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.unblock_pub = rospy.Publisher("/cmd_unblock", Bool, queue_size=1)
        self._initialpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self.box_mapper_shutdown_pub = rospy.Publisher("/box_mapper_shutdown", Bool, queue_size=1)

        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_cb, queue_size=1)
        # /amcl_pose is published only on convergence/significant change. Use /odometry/filtered
        # as a fallback so the state machine never gets stuck in INIT.
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb, queue_size=1)
        rospy.Subscriber("/mission/box_counts", Int32MultiArray, self.box_counts_cb, queue_size=1)
        rospy.Subscriber("/digit_on_nearest_box", String, self.nearest_box_digit_cb, queue_size=1)
        rospy.Subscriber("/least_frequent_digit", String, self.least_digit_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self.tick)
        rospy.loginfo("STATE_ENTER %s t=%.2f reason=boot", self.state, self.elapsed())

    def nearest_box_digit_cb(self, msg):
        self.nearest_box_digit = (msg.data or "").strip()
        self.last_nearest_box_digit_time = rospy.Time.now()


    def least_digit_cb(self, msg):
        s = (msg.data or "").strip()
        if len(s) == 1 and s.isdigit():
            self.latest_least_digit = int(s)
            self.last_least_digit_time = rospy.Time.now()

    def _parse_final_goals(self, goals_raw):
        parsed = {}
        for item in goals_raw:
            if not isinstance(item, list) or len(item) < 4:
                continue
            box_id = int(item[0])
            parsed[box_id] = [float(item[1]), float(item[2]), float(item[3])]
        return parsed

    def elapsed(self):
        return (rospy.Time.now() - self.start_time).to_sec()

    # Max allowed jump distance (m) for AMCL updates. If AMCL jumps further
    # than this from the current pose, the update is rejected.
    AMCL_MAX_JUMP = 1

    def amcl_cb(self, msg):
        self.amcl_seen = True
        cov = msg.pose.covariance
        self.amcl_pos_cov_trace = float(cov[0]) + float(cov[7])
        self._last_amcl_pose = msg.pose.pose

        # On ramp: ignore AMCL completely
        if getattr(self, "_on_ramp", False):
            return

        # Reject AMCL jumps only after ramp (not during init / 2D Pose Estimate)
        if self._jump_filter_active and self.current_pose is not None:
            dx = msg.pose.pose.position.x - self.current_pose.position.x
            dy = msg.pose.pose.position.y - self.current_pose.position.y
            jump = math.hypot(dx, dy)
            if jump > self.AMCL_MAX_JUMP:
                rospy.logwarn("AMCL jump rejected: %.2f m (max %.2f m)",
                              jump, self.AMCL_MAX_JUMP)
                # Re-seed AMCL at current position to prevent repeated jumps
                self._reinit_amcl_at_current_pose()
                return

        self.current_pose = msg.pose.pose

    def odom_cb(self, msg):
        # On ramp: always use odom
        if getattr(self, "_on_ramp", False):
            self.current_pose = msg.pose.pose
        elif self.current_pose is None:
            self.current_pose = msg.pose.pose

    def status_cb(self, msg):
        self.move_base_status = msg

    def box_counts_cb(self, msg):
        if not self.use_external_box_counts:
            return
        if not self.counting_enabled:
            return
        if len(msg.data) < 4:
            return
        self.box_counts[1] = int(msg.data[0])
        self.box_counts[2] = int(msg.data[1])
        self.box_counts[3] = int(msg.data[2])
        self.box_counts[4] = int(msg.data[3])
        self.log_box_updates_if_changed()

    def _reinit_amcl_at_current_pose(self):
        """Publish current odom pose as /initialpose with large covariance
        so AMCL re-scatters particles and re-matches from scratch using laser."""
        if self.current_pose is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose = self.current_pose
        # Small covariance so particles stay tightly clustered
        msg.pose.covariance[0] = 0.02   # var(x)
        msg.pose.covariance[7] = 0.02   # var(y)
        msg.pose.covariance[35] = 0.01  # var(yaw)
        self._initialpose_pub.publish(msg)
        rospy.loginfo("AMCL re-initialized at x=%.2f y=%.2f",
                      self.current_pose.position.x, self.current_pose.position.y)

    def _set_amcl_ramp_mode(self, on_ramp):
        """Switch between ramp mode (odom-only pose) and normal mode (AMCL pose).
        On the ramp, AMCL laser matching is unreliable because the 2D map
        does not represent the 3D slope. We bypass AMCL by using odom directly."""
        if on_ramp == self._on_ramp:
            return
        self._on_ramp = on_ramp
        if on_ramp:
            rospy.loginfo("RAMP MODE: using odometry as pose source (AMCL ignored)")
        else:
            self._reinit_amcl_at_current_pose()
            self._jump_filter_active = True
            rospy.loginfo("NORMAL MODE: AMCL jump filter activated")

    # Ramp waypoint indices where the actual 3D slope is.
    RAMP_SLOPE_START = 1
    RAMP_SLOPE_END = 2

    def transition(self, next_state, reason):
        dt = (rospy.Time.now() - self.state_enter_time).to_sec()
        rospy.loginfo("STATE_EXIT %s next=%s dt=%.2f", self.state, next_state, dt)

        # Stay on odom after ramp (do not switch back to AMCL on upper floor)

        self.state = next_state
        self.state_enter_time = rospy.Time.now()
        rospy.loginfo("STATE_ENTER %s t=%.2f reason=%s", self.state, self.elapsed(), reason)

    def send_goal(self, goal_xyz, name):
        x, y, yaw = float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])
        qx, qy, qz, qw = yaw_to_quat(yaw)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)

        self.active_goal = (x, y, yaw)
        self.active_goal_name = name
        self.goal_sent_time = rospy.Time.now()
        self.retry_count = 0
        rospy.loginfo("NAV_GOAL name=%s x=%.2f y=%.2f yaw=%.2f", name, x, y, yaw)

    def resend_active_goal(self):
        if self.active_goal is None:
            return
        x, y, yaw = self.active_goal
        self.retry_count += 1
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)
        self.goal_sent_time = rospy.Time.now()
        rospy.logwarn("NAV_RETRY name=%s retry=%d", self.active_goal_name, self.retry_count)

    def log_box_updates_if_changed(self):
        for box_id in sorted(self.box_counts.keys()):
            old_v = self._last_box_counts.get(box_id)
            new_v = self.box_counts.get(box_id)
            if old_v != new_v:
                rospy.loginfo("COUNT_UPDATE box=%d total=%d", box_id, new_v)
        self._last_box_counts = dict(self.box_counts)

    def choose_final_box(self):
        pairs = [(count, box_id) for box_id, count in self.box_counts.items()]
        pairs.sort(key=lambda x: (x[0], x[1]))
        return pairs[0][1]

    def distance_to_active_goal(self):
        if self.current_pose is None or self.active_goal is None:
            return None
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        gx, gy, _ = self.active_goal
        return math.hypot(gx - x, gy - y)

    def distance_to_point(self, point_xyz):
        if self.current_pose is None:
            return None
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        gx = float(point_xyz[0])
        gy = float(point_xyz[1])
        return math.hypot(gx - x, gy - y)

    def active_goal_reached(self):
        d = self.distance_to_active_goal()
        if d is None or d >= self.goal_tolerance:
            return False
        if self.yaw_tolerance <= 0 or self.current_pose is None or self.active_goal is None:
            return True
        q = self.current_pose.orientation
        cur_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        tgt_yaw = float(self.active_goal[2])
        dyaw = math.atan2(math.sin(tgt_yaw - cur_yaw), math.cos(tgt_yaw - cur_yaw))
        return abs(dyaw) < self.yaw_tolerance

    def active_goal_timed_out(self):
        if self.goal_sent_time is None:
            return False
        return (rospy.Time.now() - self.goal_sent_time).to_sec() > self.goal_timeout

    def start_route(self, points, route_name):
        self.route = points
        self.route_idx = 0
        if len(self.route) == 0:
            rospy.logwarn("Route %s is empty", route_name)
            return
        self.send_goal(self.route[self.route_idx], "%s_%d" % (route_name, self.route_idx))

    def _move_base_is_idle(self):
        """Check if move_base has no active goal (status list empty or all succeeded/aborted)."""
        if self.move_base_status is None:
            return False
        for s in self.move_base_status.status_list:
            if s.status in (0, 1):  # 0=PENDING, 1=ACTIVE
                return False
        return True

    def route_step(self, route_name):
        if len(self.route) == 0:
            return True
        if self.active_goal_reached():
            self.route_idx += 1
            if self.route_idx >= len(self.route):
                return True
            self.send_goal(self.route[self.route_idx], "%s_%d" % (route_name, self.route_idx))
            return False

        # If move_base stopped (GOAL Reached! / aborted) but we haven't reached, resend
        if self._move_base_is_idle() and self.goal_sent_time is not None:
            elapsed = (rospy.Time.now() - self.goal_sent_time).to_sec()
            if elapsed > 2.0:  # wait 2s before resending to avoid spam
                rospy.logwarn("move_base idle but goal not reached, resending %s_%d",
                              route_name, self.route_idx)
                self.resend_active_goal()

        if self.active_goal_timed_out():
            if self.retry_count < self.max_retries:
                self.resend_active_goal()
            else:
                rospy.logerr("NAV_FAIL route=%s idx=%d timeout", route_name, self.route_idx)
                self.transition("FAIL", "route_timeout")
        return False

    def _amcl_converged(self):
        """Return True if we either don't care about AMCL, or AMCL covariance is small."""
        if not self.wait_for_amcl:
            return True
        if not self.amcl_seen or self.amcl_pos_cov_trace is None:
            return False
        return self.amcl_pos_cov_trace < self.amcl_cov_threshold

    def tick(self, _):
        if self.state == "INIT":
            ready = (self.current_pose is not None) and (self.move_base_status is not None)
            init_dt = (rospy.Time.now() - self.state_enter_time).to_sec()

            amcl_ok = self._amcl_converged()
            if ready and not amcl_ok and init_dt < self.amcl_max_wait:
                # Not ready yet — log every ~2s so user knows what's blocking us.
                if int(init_dt * 0.5) != getattr(self, "_last_log_sec", -1):
                    self._last_log_sec = int(init_dt * 0.5)
                    if not self.amcl_seen:
                        rospy.loginfo_throttle(2.0,
                            "INIT: waiting for AMCL pose (none received yet) "
                            "— give a 2D Pose Estimate in RViz")
                    else:
                        rospy.loginfo_throttle(2.0,
                            "INIT: waiting for AMCL convergence "
                            "(cov_trace=%.3f, threshold=%.3f)",
                            self.amcl_pos_cov_trace, self.amcl_cov_threshold)
                return

            if ready and init_dt >= self.init_wait:
                if init_dt >= self.amcl_max_wait and not amcl_ok:
                    rospy.logwarn(
                        "INIT: AMCL did not converge within %.1fs (cov_trace=%s); "
                        "starting mission anyway",
                        self.amcl_max_wait,
                        ("%.3f" % self.amcl_pos_cov_trace) if self.amcl_pos_cov_trace else "n/a")
                self.transition("SCAN_LOWER", "amcl_and_move_base_ready")
                self.start_route(self.lower_waypoints, "lower")
            return

        if self.state == "SCAN_LOWER":
            self.log_box_updates_if_changed()
            done = self.route_step("lower")
            last_point_reached = False
            if len(self.lower_waypoints) > 0 and self.lower_unlock_on_last_point:
                d_last = self.distance_to_point(self.lower_waypoints[-1])
                last_point_reached = (
                    d_last is not None and d_last < self.lower_last_point_tolerance
                )

            if done or last_point_reached:
                if self.latest_least_digit is None:
                    rospy.logwarn("No /least_frequent_digit received yet; stay in SCAN_LOWER")
                    return

                self.target_digit = self.latest_least_digit
                self.counting_enabled = False

                rospy.loginfo(
                    "TARGET_DIGIT_DECIDED digit=%d",
                    self.target_digit,
                )
                self.counting_enabled = False

                rospy.loginfo(
                    "TARGET_DIGIT_DECIDED digit=%d count=%d",
                    self.target_digit,
                    self.box_counts.get(self.target_digit, -1),
                )

                if last_point_reached and not done:
                    rospy.loginfo(
                        "LOWER_LAST_POINT_REACHED d=%.2f tol=%.2f",
                        self.distance_to_point(self.lower_waypoints[-1]),
                        self.lower_last_point_tolerance,
                    )

                self.transition("UNBLOCK", "target_digit_decided_before_unblock")
            return

        if self.state == "UNBLOCK":
            if not self.unblock_sent:
                self.unblock_pub.publish(Bool(data=True))
                self.unblock_sent = True
                rospy.loginfo("UNBLOCK_SENT t=%.2f", self.elapsed())
            unblock_dt = (rospy.Time.now() - self.state_enter_time).to_sec()
            # Wait 2 seconds for cone to fully disappear and costmap to clear
            if unblock_dt >= 2.0:
                if self.stop_after_unblock:
                    self.transition("DONE", "stop_after_unblock")
                else:
                    # Clear costmaps so the cone's ghost doesn't block planning
                    try:
                        rospy.ServiceProxy("/move_base/clear_costmaps", std_srvs.srv.Empty)()
                        rospy.loginfo("Costmaps cleared after unblock")
                    except Exception:
                        pass
                    self.transition("GO_EXIT", "unblock_sent")
                    self.send_goal(self.exit_goal, "exit_goal")
            return

        if self.state == "GO_EXIT":
            if not self.box_mapper_shutdown_sent:
                self.box_mapper_shutdown_pub.publish(Bool(data=True))
                self.box_mapper_shutdown_sent = True
                rospy.loginfo("BOX_MAPPER_SHUTDOWN_SENT")

            if self.active_goal_reached():
                self.transition("GO_RAMP", "exit_reached")
                self.start_route(self.ramp_waypoints, "ramp")
                return

            if self.active_goal_timed_out():
                if self.retry_count < self.max_retries:
                    self.resend_active_goal()
                else:
                    self.transition("FAIL", "exit_timeout")
            return

        if self.state == "GO_RAMP":
            done = self.route_step("ramp")
            if done:
                self.door_ids_in_order = sorted(self.door_observation_goals.keys())
                self.current_door_search_idx = 0

                if len(self.door_ids_in_order) == 0:
                    self.transition("FAIL", "no_door_observation_goals")
                    return

                first_door_id = self.door_ids_in_order[self.current_door_search_idx]
                self.transition("SEARCH_DOOR", "ramp_done")
                self.send_goal(self.door_observation_goals[first_door_id], "door_obs_%d" % first_door_id)
            return

        if self.state == "SEARCH_DOOR":
            if self.active_goal_reached():
                if 0 <= self.current_door_search_idx < len(self.door_ids_in_order):
                    self.target_door_id = self.door_ids_in_order[self.current_door_search_idx]
                    self.transition("VERIFY_DOOR_DIGIT", "door_observation_reached")
                    return

            if self.active_goal_timed_out():
                if self.retry_count < self.max_retries:
                    self.resend_active_goal()
                else:
                    self.transition("FAIL", "door_observation_timeout")
            return

            if done:
                self.transition("FAIL", "no_matching_door_found")
            return

        if self.state == "VERIFY_DOOR_DIGIT":
            verify_dt = (rospy.Time.now() - self.state_enter_time).to_sec()
            target_digit_str = str(self.target_digit)
            seen_digit = ""
            if self.last_nearest_box_digit_time is not None:
                age = (rospy.Time.now() - self.last_nearest_box_digit_time).to_sec()
                if age < 1.0:
                    seen_digit = self.nearest_box_digit

            if seen_digit == target_digit_str:
                rospy.loginfo(
                    "TARGET_DOOR_FOUND door=%d target_digit=%s seen_digit=%s",
                    self.target_door_id, target_digit_str, seen_digit
                )
                self.transition("ENTER_TARGET_DOOR", "door_digit_matched")
                self.send_goal(
                    self.door_entry_goals[self.target_door_id],
                    "door_entry_%d" % self.target_door_id
                )
                return

            if verify_dt >= self.verify_wait:
                rospy.loginfo(
                    "DOOR_DIGIT_MISMATCH door=%d target_digit=%s seen_digit=%s",
                    self.target_door_id, target_digit_str, seen_digit
                )

                self.current_door_search_idx += 1
                if self.current_door_search_idx < len(self.door_ids_in_order):
                    next_door_id = self.door_ids_in_order[self.current_door_search_idx]
                    self.transition("SEARCH_DOOR", "try_next_door")
                    self.send_goal(
                        self.door_observation_goals[next_door_id],
                        "door_obs_%d" % next_door_id
                    )
                else:
                    self.transition("FAIL", "no_matching_door_found")
                return

            return

        if self.state == "ENTER_TARGET_DOOR":
            if self.active_goal_reached():
                self.transition("DONE", "target_door_entered")
                return

            if self.active_goal_timed_out():
                if self.retry_count < self.max_retries:
                    self.resend_active_goal()
                else:
                    self.transition("FAIL", "target_door_entry_timeout")
            return


if __name__ == "__main__":
    rospy.init_node("mission_manager")
    MissionManager()
    rospy.spin()
