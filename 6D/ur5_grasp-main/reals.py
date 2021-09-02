import time
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import numpy as np
import cv2
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from inference import test
import threading

# camera config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #配置depth流
align_to = rs.stream.color  #与color流对齐
align = rs.align(align_to)

pipeline.start(config)

# model config
modelcfg   = 'cfg/MFPN-yolov4-pose.cfg'

weightfile = 'backup/duck/MFPN.weights'

# ur5 config
rob = urx.Robot("192.168.12.3")
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
a = 0.1
v = 0.1

# matrix from camera to robot
cameratorobot = np.array([[ 0.00748487, -0.82486757,  0.56527646, -1.0575007], [-0.99968723, 0.0073173,  0.02391455, -0.14602299], [-0.02386263, -0.56527865,  -0.8245548, 0.92100175], [0,0,0,1]])


lock = threading.Lock()


def move_robot(color_image):
    if lock.acquire():
        try:
            x,y = test(modelcfg, weightfile, color_image)
            print("image_x = ", x, ",image_y = ", y)
            dis = aligned_depth_frame.get_distance(x, y)  #（x, y)点的真实深度值
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  #（x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。

            print("camera_x = ", camera_coordinate[0] , ",camera_y = ",camera_coordinate[1], ",camera_z = ", camera_coordinate[2])

            camera_coordinate = np.append(camera_coordinate, 1)
            import pdb
            pdb.set_trace()
            C = np.dot(cameratorobot, camera_coordinate)
            rob.movep((C[0], C[1], C[2]-0.03, 2.3248, -2.2405, -0.0507),a ,v)
            robotiqgrip.close_gripper()
            rob.movep((-0.341, -0.380, 0.257, 2.75, -1.63, 0.014),a, v)
            robotiqgrip.open_gripper()
            print("x = ", C[0] , ",y = ",C[1], ",z = ", C[2])
            print("\n")
            print("\n")

        finally:
            lock.release()

    print("ok")
    

    return "ok"

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  #获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()   #获取对齐帧中的color帧
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  #获取深度参数（像素坐标系转相机坐标系会用到）

        # color_frame = frames.get_color_frame()

    
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.asanyarray(color_frame.get_data())
        

        t = threading.Thread(target=move_robot, args=(color_image,))
        t.start()
        # time.sleep(2)
        x,y = test(modelcfg, weightfile, color_image)
        

        result_image = cv2.circle(color_image, (int(x),int(y)), 3, (0,0,255), 0)
        
        cv2.imwrite('save.jpeg',color_image)
        cv2.imshow('RealSense', result_image)
        cv2.waitKey(1)
finally:
    pipeline.stop()