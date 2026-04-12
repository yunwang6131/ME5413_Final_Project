# ROS Box Mapper 启动与调试指南

## 一、启动流程

### 终端1：启动仿真环境
```bash
cd ~/ME5413_Final_Project
source devel/setup.bash
roslaunch me5413_world world.launch
```


### 终端2：重生场景物体
```bash
cd ~/ME5413_Final_Project
source devel/setup.bash
rostopic pub -1 /rviz_panel/respawn_objects std_msgs/Int16 "data: 1"
```

建议等待仿真完全启动后再执行

### 终端3：启动导航与 Box Mapper
```bash
cd ~/ME5413_Final_Project
source devel/setup.bash
roslaunch me5413_world navigation.launch use_teb:=true
'''

rostopic echo /clicked_point
rostopic echo /rosout | grep --line-buffered -E "STATE_ENTER|STATE_EXIT|NAV_GOAL|UNBLOCK_SENT|NAV_FAIL|NAV_RETRY|FINAL_DECISION|DOOR_DIGIT|TARGET_DOOR"


cd ~/ME5413_Final_Project
source devel/setup.bash
rostopic echo /rosout | grep --line-buffered -E "COUNT_UPDATE|TARGET_DIGIT_DECIDED"

cd ~/ME5413_Final_Project
source devel/setup.bash
rostopic echo /least_frequent_digit

killall -9 roscore
killall -9 rosmaster
killall -9 gzserver
killall -9 gzclient