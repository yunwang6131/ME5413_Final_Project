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
        self.state_enter_time = rospy.Time.now() #记录进入当前状态的时间
        self.start_time = rospy.Time.now() #记录整个任务启动的时间

        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.45)) #从 ROS 参数服务器读 goal_tolerance，如果没有就默认 0.45表示到目标点多近算“到达”。
        self.yaw_tolerance = float(rospy.get_param("~yaw_tolerance", 0.2)) # 读取朝向误差容忍度，默认 0.2 弧度
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 140.0)) # 单个目标导航超时时间，默认 60 秒
        self.max_retries = int(rospy.get_param("~max_retries", 2)) #目标失败后最多重发几次，默认 2 次。
        self.tick_hz = float(rospy.get_param("~tick_hz", 2.0)) #状态机更新频率，默认 2.0 Hz
        self.init_wait = float(rospy.get_param("~init_wait", 2.0)) #初始化阶段最少等待 2 秒
        self.unblock_wait = float(rospy.get_param("~unblock_wait", 1.0)) #解锁阶段的等待时间参数，真正用的是写死的 2 秒，不是这里这个变量
        self.stop_after_unblock = bool(rospy.get_param("~stop_after_unblock", False)) #如果设成 True，解锁后就不继续任务，直接结束
        self.lower_unlock_on_last_point = bool(
            rospy.get_param("~lower_unlock_on_last_point", True)
        ) #是否在下层路线最后一个点附近就提前判定完成扫描
        self.lower_last_point_tolerance = float(
            rospy.get_param("~lower_last_point_tolerance", 0.60)
        )  # 下层路线最后一个点的特殊容差默认 0.60

        self.wait_for_amcl = bool(rospy.get_param("~wait_for_amcl", True)) #任务开始前，是否等待 AMCL 收敛后再开始任务  
        self.amcl_cov_threshold = float(rospy.get_param("~amcl_cov_threshold", 0.5)) #AMCL 收敛阈值，协方差迹小于它就认为定位可信。
        self.amcl_max_wait = float(rospy.get_param("~amcl_max_wait", 30.0)) #最多等 AMCL 30 秒，超过后即使没收敛也会继续任务。
        self.amcl_pos_cov_trace = None #当前 AMCL 的协方差迹，用于判断收敛状态。
        self.amcl_seen = False #是否已经收到过 AMCL 消息
        self._on_ramp = False #是否在坡道上，此时不使用 AMCL
        self._last_amcl_pose = None #上一次 AMCL 的位姿，用于计算跳变
        self._jump_filter_active = False #是否激活跳变过滤器

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
        self.door_observation_goals = self._parse_final_goals(self.door_observation_goals_raw) #把原始列表转成字典，格式变成：{1:[x,y,yaw], 2:[x,y,yaw], ...}这样后面按门 ID 查目标更方便

        self.door_entry_goals_raw = rospy.get_param(
            "~door_entry_goals",
            [
                [1, 3.4, -7.59, 3.14],
                [2, 3.4, -2.47, 3.14],
                [3, 3.4,  2.57, 3.14],
                [4, 3.4,  7.75, 3.14],
            ],
        )
        self.door_entry_goals = self._parse_final_goals(self.door_entry_goals_raw) #同样转成字典方便按 ID 索引。

        self.current_pose = None
        self.move_base_status = None
        self.active_goal = None  #当前正在执行的目标点 (x, y, yaw)
        self.active_goal_name = "" #当前目标的名字，比如 lower_0
        self.goal_sent_time = None #当前目标发送的时间，用来判断是否超时
        self.retry_count = 0 #当前目标重试次数

        self.route = [] #当前正在走的整条路径
        self.route_idx = -1 #当前走到路径的第几个点
        self.unblock_sent = False #是否已经发送过解锁信号

        self.target_digit = None #最终选中的目标数字
        self.target_door_id = None #最终决定要进入的门 ID
        self.observed_door_digits = {}   # 将观察到的门和门内的数字存成字典
        self.door_arrival_time = None          # 当前门真正到达的时刻
        self.current_observation_door_id = None  # 当前正在等待观察的门 ID
        self.current_door_seen_digits = []   # 当前门观察窗口内收到的所有非空数字
        self.last_target_digit_time = None

        self.counting_enabled = True #是否允许继续统计箱子数，扫描完成后会关掉，避免后面数据再变化。
        self.box_mapper_shutdown_sent = False #是否已经发送过关闭 box_mapper 的信号

        self.door_ids_in_order = [] #门的检查顺序列表
        self.current_door_search_idx = -1 #当前正在看的门的索引

        self.nearest_box_digit = "" #当前离机器人最近的箱子数字
        self.last_nearest_box_digit_time = None #上次收到最近箱子数字的时间
        self.min_observe_wait = float(rospy.get_param("~min_observe_wait", 20.0))
        self.max_observe_wait = float(rospy.get_param("~max_observe_wait", 100.0))

        self.red_detected = False              # /red_detected 当前持续状态
        self.red_left_event = False            # /red_left 事件标志，只在收到一次 True 后置位
        self.last_red_left_time = None         # 最近一次收到 /red_left=True 的时刻
        self.seen_red_after_arrival = False    # 到达目标门观察点后，是否至少看到过一次红色
        self.target_door_observation_arrival_time = None
        self.red_wait_timeout = float(rospy.get_param("~red_wait_timeout", 20.0))
        

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1) #创建发布器，向 /move_base_simple/goal 发送目标点
        self.unblock_pub = rospy.Publisher("/cmd_unblock", Bool, queue_size=1)
        self._initialpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1) #创建发布器，向 /initialpose 发送初始位姿
        self.box_mapper_shutdown_pub = rospy.Publisher("/box_mapper_shutdown", Bool, queue_size=1)

        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_cb, queue_size=1) #订阅 /amcl_pose，回调函数是 amcl_cb
        # /amcl_pose is published only on convergence/significant change. Use /odometry/filtered
        # as a fallback so the state machine never gets stuck in INIT.
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_cb, queue_size=1) #订阅 /odometry/filtered，回调函数是 odom_cb
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.status_cb, queue_size=1) #订阅 /move_base/status，回调函数是 status_cb
        rospy.Subscriber("/least_frequent_digit", String, self.least_frequent_digit_cb, queue_size=1)
        rospy.Subscriber("/digit_on_nearest_box", String, self.nearest_box_digit_cb, queue_size=1) #订阅 /digit_on_nearest_box，回调函数是 nearest_box_digit_cb
        
        rospy.Subscriber("/red_detected", Bool, self.red_detected_cb, queue_size=1)
        rospy.Subscriber("/red_left", Bool, self.red_left_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self.tick) #创建定时器，每隔 1/tick_hz 秒执行一次 tick()
        rospy.loginfo("STATE_ENTER %s t=%.2f reason=boot", self.state, self.elapsed()) #记录进入状态和启动时间

    def nearest_box_digit_cb(self, msg):
        self.nearest_box_digit = (msg.data or "").strip()
        self.last_nearest_box_digit_time = rospy.Time.now()

    def least_frequent_digit_cb(self, msg):
        "收到 /least_frequent_digit 消息，直接保存成 self.target_digit"
        data = (msg.data or "").strip()
        if data == "":
            return
        self.target_digit = data
        self.last_target_digit_time = rospy.Time.now()

    def red_detected_cb(self, msg):
        self.red_detected = bool(msg.data)

    def red_left_cb(self, msg):
        # /red_left 是下降沿事件，不是持续状态
        if bool(msg.data):
            self.red_left_event = True
            self.last_red_left_time = rospy.Time.now()

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


    def transition(self, next_state, reason):
        dt = (rospy.Time.now() - self.state_enter_time).to_sec()
        if self.state == "SEARCH_DOOR":
            rospy.loginfo(
                "STATE_EXIT %s next=%s dt=%.2f observed=%s reason=%s",
                self.state, next_state, dt, str(self.observed_door_digits), reason
            )
        else:
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
            done = self.route_step("lower")
            last_point_reached = False

            if len(self.lower_waypoints) > 0 and self.lower_unlock_on_last_point:
                d_last = self.distance_to_point(self.lower_waypoints[-1])
                last_point_reached = (
                    d_last is not None and d_last < self.lower_last_point_tolerance
                )

            if done or last_point_reached:
                if self.target_digit is None or self.target_digit == "":
                    rospy.logwarn("SCAN_LOWER finished but /least_frequent_digit not received yet")
                    return

                self.counting_enabled = False

                rospy.loginfo(
                    "TARGET_DIGIT_DECIDED digit=%s",
                    self.target_digit,
                )

                if last_point_reached and not done:
                    rospy.loginfo(
                        "LOWER_LAST_POINT_REACHED d=%.2f tol=%.2f",
                        self.distance_to_point(self.lower_waypoints[-1]),
                        self.lower_last_point_tolerance,
                    )

                self.transition("UNBLOCK", "target_digit_received_before_unblock")
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
                self.observed_door_digits = {}
                self.target_door_id = None
                self.door_arrival_time = None
                self.current_observation_door_id = None
                self.current_door_seen_digits = []

                if len(self.door_ids_in_order) == 0:
                    self.transition("FAIL", "no_door_observation_goals")
                    return

                first_door_id = self.door_ids_in_order[self.current_door_search_idx] #然后发第一个门的观察点目标
                self.transition("SEARCH_DOOR", "ramp_done")
                self.send_goal(self.door_observation_goals[first_door_id], "door_obs_%d" % first_door_id)
            return

        if self.state == "SEARCH_DOOR":
            if self.active_goal_reached():
                if 0 <= self.current_door_search_idx < len(self.door_ids_in_order):
                    door_id = self.door_ids_in_order[self.current_door_search_idx]

                    # 初始化采样环境
                    if self.current_observation_door_id != door_id:
                        self.current_observation_door_id = door_id
                        self.door_arrival_time = rospy.Time.now()
                        self.current_door_seen_digits = []
                        self.nearest_box_digit = ""
                        self.last_nearest_box_digit_time = None

                        rospy.loginfo(
                            "ARRIVED_AT_DOOR door_id=%d, start observing, min_wait=%.1f max_wait=%.1f",
                            door_id, self.min_observe_wait, self.max_observe_wait
                        )
                        return

                    observe_dt = (rospy.Time.now() - self.door_arrival_time).to_sec()

                    # 只收集“到达这扇门之后”收到的新识别结果
                    if (
                        self.last_nearest_box_digit_time is not None
                        and self.last_nearest_box_digit_time >= self.door_arrival_time
                    ):
                        seen_digit = (self.nearest_box_digit or "").strip()
                        if seen_digit != "":
                            self.current_door_seen_digits.append(seen_digit)
                            rospy.loginfo(
                                "DOOR %d NEW_SAMPLE digit=%s samples=%s",
                                door_id, seen_digit, str(self.current_door_seen_digits)
                            )
                            self.nearest_box_digit = ""

                    # 不到最小观察时间，绝不离开
                    if observe_dt < self.min_observe_wait:
                        return

                    # 统计当前已收集结果
                    if len(self.current_door_seen_digits) > 0:
                        counts = {}
                        for d in self.current_door_seen_digits:
                            counts[d] = counts.get(d, 0) + 1
                        seen_digit = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                    else:
                        seen_digit = ""

                    # 满足最小观察时间后，只要收到结果就立刻走
                    if seen_digit != "":
                        self.observed_door_digits[door_id] = seen_digit
                        rospy.loginfo(
                            "DOOR %d RESULT: %s (samples: %s)",
                            door_id, seen_digit, self.current_door_seen_digits
                        )
                        rospy.loginfo(
                            "DOOR_OBSERVED door=%d final_digit=%s raw=%s observed=%s",
                            door_id, seen_digit, str(self.current_door_seen_digits), str(self.observed_door_digits)
                        )
                    else:
                        # 没收到结果时，最多等到 max_observe_wait
                        if observe_dt < self.max_observe_wait:
                            return

                        self.observed_door_digits[door_id] = ""
                        rospy.logwarn(
                            "DOOR %d RESULT EMPTY after %.1f s (samples: %s)",
                            door_id, observe_dt, str(self.current_door_seen_digits)
                        )
                        rospy.loginfo(
                            "DOOR_OBSERVED door=%d final_digit=%s raw=%s observed=%s",
                            door_id, "", str(self.current_door_seen_digits), str(self.observed_door_digits)
                        )

                    # 继续看下一扇门
                    self.current_door_search_idx += 1
                    self.current_observation_door_id = None
                    self.door_arrival_time = None
                    self.nearest_box_digit = ""
                    self.current_door_seen_digits = []

                    if self.current_door_search_idx < len(self.door_ids_in_order):
                        next_door_id = self.door_ids_in_order[self.current_door_search_idx]
                        self.send_goal(
                            self.door_observation_goals[next_door_id],
                            "door_obs_%d" % next_door_id
                        )
                        return
                    else:
                        self.transition("VERIFY_DOOR_DIGIT", "all_doors_observed")
                        return

            if self.active_goal_timed_out():
                if self.retry_count < self.max_retries:
                    self.resend_active_goal()
                else:
                    self.transition("FAIL", "door_observation_timeout")
            return

        if self.state == "VERIFY_DOOR_DIGIT":
            rospy.loginfo("OBSERVED_DOOR_DIGITS = %s", self.observed_door_digits)
            target_digit_str = (self.target_digit or "").strip()

            matched_doors = [
                door_id for door_id, digit in self.observed_door_digits.items()
                if digit == target_digit_str
            ]

            rospy.loginfo(
                "FINAL_COMPARE target_digit=%s observed=%s matched_doors=%s",
                target_digit_str, str(self.observed_door_digits), str(matched_doors)
            )

            if len(matched_doors) == 0:
                self.transition("FAIL", "no_matching_door_found")
                return
            # 如果有多个匹配，默认选 door_id 最小的
            self.target_door_id = sorted(matched_doors)[0]

            rospy.loginfo(
                "TARGET_DOOR_FOUND door=%d target_digit=%s",
                self.target_door_id, target_digit_str
            )

            # 准备回到目标门观察点，重新等待红色动态障碍通过
            self.red_left_event = False
            self.seen_red_after_arrival = False
            self.target_door_observation_arrival_time = None

            self.transition("GO_TARGET_DOOR_OBSERVATION", "door_digit_matched")
            self.send_goal(
                self.door_observation_goals[self.target_door_id],
                "target_door_obs_%d" % self.target_door_id
            )
            return

        if self.state == "GO_TARGET_DOOR_OBSERVATION":
            if self.active_goal_reached():
                self.target_door_observation_arrival_time = rospy.Time.now()
                self.red_left_event = False
                self.seen_red_after_arrival = False

                rospy.loginfo(
                    "ARRIVED_TARGET_DOOR_OBSERVATION door=%d, waiting for red obstacle to pass",
                    self.target_door_id
                )

                self.transition("WAIT_RED_CLEAR", "target_door_observation_reached")
                return

            if self.active_goal_timed_out():
                if self.retry_count < self.max_retries:
                    self.resend_active_goal()
                else:
                    self.transition("FAIL", "target_door_observation_timeout")
            return
        
        if self.state == "WAIT_RED_CLEAR":
            if self.target_door_observation_arrival_time is None:
                self.target_door_observation_arrival_time = rospy.Time.now()

            wait_dt = (rospy.Time.now() - self.target_door_observation_arrival_time).to_sec()

            # 到达目标门观察点之后，只要看见过红色一次，就记住
            if self.red_detected:
                if not self.seen_red_after_arrival:
                    rospy.loginfo("RED_OBSTACLE_SEEN door=%d", self.target_door_id)
                self.seen_red_after_arrival = True

            # 必须先看到过红色，再等红色离开事件
            if self.seen_red_after_arrival and self.red_left_event:
                rospy.loginfo(
                    "RED_OBSTACLE_LEFT door=%d, entering now",
                    self.target_door_id
                )
                self.red_left_event = False
                self.transition("ENTER_TARGET_DOOR", "red_left_after_seen")
                self.send_goal(
                    self.door_entry_goals[self.target_door_id],
                    "door_entry_%d" % self.target_door_id
                )
                return

            if wait_dt > self.red_wait_timeout:
                rospy.logwarn(
                    "WAIT_RED_CLEAR timeout door=%d seen_red=%s red_detected=%s",
                    self.target_door_id,
                    str(self.seen_red_after_arrival),
                    str(self.red_detected)
                )
                self.transition("FAIL", "wait_red_clear_timeout")
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
