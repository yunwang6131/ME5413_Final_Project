## 指令合集

## 非正常关闭 Gazebo/ROS

```bash
killall -9 roscore
killall -9 rosmaster
killall -9 gzserver
killall -9 gzclient
```
## 启动仿真世界
```bash
source devel/setup.bash
roslaunch me5413_world world.launch
```
## 重生物体（RViz 面板）
```bash
source devel/setup.bash
rostopic pub -1 /rviz_panel/respawn_objects std_msgs/Int16 "data: 1"
```
## 启动导航
```bash
source devel/setup.bash
roslaunch me5413_world navigation.launch use_teb:=true
```
## 常用调试 Topic
### 查看当前最小数字

```bash
rostopic echo /least_frequent_digit
```
### 查看当前最近箱子的数字
```bash
rostopic echo /digit_on_nearest_box
```
### 查看下一步目标点坐标
```bash
rostopic echo /move_base_simple/goal
```
### 查看当前导航状态（是否到达）
```bash
rostopic echo /move_base/status
```
### 查看状态机关键日志
```bash
rostopic echo /rosout | grep --line-buffered -E "STATE_ENTER|STATE_EXIT|NAV_GOAL|UNBLOCK_SENT|NAV_FAIL|NAV_RETRY|FINAL_DECISION"
```
## 点击点位相关
### 查看 RViz 点击点
```bash
rostopic echo /clicked_point
```
## 红色障碍相关
### 查看红色相关节点是否启动
```bash
rostopic list | grep red
```
正常应包含：
- `/red_detected`
- `/red_left`
### 持续状态（是否检测到红色）
```bash
rostopic echo /red_detected
```
输出示例：
```text
data: True
data: False
```
### 离开事件（关键触发）
```bash
rostopic echo /red_left
```
输出示例：
```text
data: True
```