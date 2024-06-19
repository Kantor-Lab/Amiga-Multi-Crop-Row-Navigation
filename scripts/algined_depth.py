#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class DepthColorAligner:
    def __init__(self):
        rospy.init_node('depth_color_aligner')

        # Initialize the bridge between ROS and OpenCV
        self.bridge = CvBridge()

        # Create ROS publisher for aligned depth image
        self.depth_aligned_pub = rospy.Publisher('/aligned_depth_to_color', Image, queue_size=10)

        # Create ROS subscribers for color and depth images
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_image_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_image_callback)

        # Initialize color intrinsics
        self.color_intrinsics = None

    def color_image_callback(self, data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.color_intrinsics = color_image.shape
        except Exception as e:
            rospy.logerr(e)

    def depth_image_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, 'passthrough')

            # Resize depth image to match color image resolution
            depth_image_resized = cv2.resize(depth_image, (self.color_intrinsics[1], self.color_intrinsics[0]))

            # Publish the aligned depth image
            depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image_resized, 'passthrough')
            depth_image_msg.header = data.header
            self.depth_aligned_pub.publish(depth_image_msg)
        except Exception as e:
            rospy.logerr(e)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        aligner = DepthColorAligner()
        aligner.run()
    except rospy.ROSInterruptException:
        pass
