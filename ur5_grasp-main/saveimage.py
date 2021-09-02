import pyrealsense2 as rs
import numpy as np

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #配置depth流

align_to = rs.stream.color  #与color流对齐
align = rs.align(align_to)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  #获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()   #获取对齐帧中的color帧
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  #获取深度参数（像素坐标系转相机坐标系会用到）
        depth_frame = aligned_frames.get_depth_frame()

    
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cv2.imwrite('depth.jpeg',depth_image)
        cv2.imwrite('save.jpeg',color_image)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()