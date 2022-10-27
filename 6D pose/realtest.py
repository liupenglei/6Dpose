import time
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import numpy as np
import cv2
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from inference import test
from inference import test1
import threading
import math

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
a = 0.15
v = 0.15
# matrix from camera to robot


cameratorobot = np.array([[-0.0021041,  -0.81133705,  0.58457486, -0.90946656],
 [-0.99932431, -0.01974484, -0.03100098, 0.12146555],
 [ 0.03669458, -0.5842451,  -0.81074729,  0.89359417],
 [ 0. ,         0.   ,       0.  ,        1.  ,      ]])



cameratorobot_r = np.array([[-0.00600037, -0.92703532,  0.374926],
 [-0.99908734,  0.02141406,  0.03695849],
 [-0.04229052, -0.37436205, -0.92631775]])


lock = threading.Lock()


def move_robot_v2(x,y,confi,rot):
    if lock.acquire():
        try:
            dis = aligned_depth_frame.get_distance(x, y)  #（x, y)点的真实深度值
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  #（x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
            
            Rot = np.dot(cameratorobot_r, rot)

            def RE(R):
                sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
                singular = sy < 1e-6

                if not singular:
                    x = math.atan2(R[2,1],R[2,2])
                    y = math.atan2(-R[2,0],sy)
                    z = math.atan2(R[1,0],R[0,0])
                else:
                    x = math.atan2(-R[1,2],R[1,1])
                    y = math.atan2(-R[2,0],sy)
                    z = 0
                return np.array([x,y,z])
            
            c = RE(Rot)

            # print("camera_x = ", camera_coordinate[0] , ",camera_y = ",camera_coordinate[1], ",camera_z = ", camera_coordinate[2])
            # print("R :",c[2],c[0],c[1])

        
            camera_coordinate = np.append(camera_coordinate, 1)
            
            C = np.dot(cameratorobot, camera_coordinate)

            rob.movel((-0.22018, -0.11911, 0.27 , 2.27, -2.18, 0.0148),a, v)

            rob.movep((C[0], C[1], C[2]+0.22, 3.13, -0.65, -0.077),a ,v)

            

            if confi > 0.999:
                #rob.movep((C[0]+0.005, C[1], C[2] + 0.015 , 2.3*0.75+0.15*c[2], -2.24*0.75+0.15*c[0], -0.05*0.6+0.4*c[1]),a ,v)
                rob.movep((C[0]+0.016, C[1], C[2]-0.03  , c[2], c[0], c[1]),a ,v)
            
            else:
                rob.movep((C[0]+0.016, C[1], C[2]-0.03 , 2.3, -2.2, -0.0057),a ,v)



            #rob.movep((C[0], C[1], C[2]+0, 0.3*c[2]+0.7*2.3, -2.24*0.7+0.3*c[0], 0.05),a ,v)
            robotiqgrip.close_gripper()

            rob.movep((C[0]+0.015, C[1], C[2]+0.2, 2.3248, -2.2405, -0.0507),a ,v)
            

            rob.movel((0.388, -0.4325, 0.335 , 3.1, 0.4877, 0.0058),a, v)

            rob.movel((0.388, -0.4325, C[2] , 3.1, 0.4877, 0.0058),a, v)

           
            robotiqgrip.open_gripper()

            
            rob.movel((0.388, -0.4325, 0.335 , 3.1, 0.4877, 0.0058),a, v)

    
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

    
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.asanyarray(color_frame.get_data())
        

        t = threading.Thread(target=move_robot_v2, args=(color_image,))
        t.start()
        # time.sleep(2)
        x,y,confi,rot = test(modelcfg, weightfile, color_image)

        if (x==0 and y==0):
            continue
        else:
            move_robot_v2(x,y,confi,rot)
        

        result_image = cv2.circle(color_image, (int(x),int(y)), 3, (0,0,255), 0)
        
        #cv2.imwrite('save.jpeg',color_image)
        cv2.imshow('RealSense', result_image)
        cv2.waitKey(1)
finally:
    pipeline.stop()
