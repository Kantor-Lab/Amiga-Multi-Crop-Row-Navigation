#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
import numpy as np

def angle_callback(source_msg):
    # Create a new message of the desired type
    

    # Perform the necessary conversion logic
    # Here you will extract the relevant data from the source_msg
    # and assign it to the corresponding fields in the converted_msg

    # Assign values to the converted_msg fields
    x = source_msg.linear_acceleration.x
    angle = np.arccos(x/-9.81)
    angle = angle * (180 / np.pi)
    # Publish the converted message on the new topic
    # pub.publish(converted_msg)
    print("tilted angle in degree:", angle)

if __name__ == '__main__':
    rospy.init_node('angle_calculate')

    # Create a publisher for the converted messages
    # pub = rospy.Publisher('/gps', PoseWithCovarianceStamped, queue_size=10)

    # Create a subscriber to the original topic with the source message type
    rospy.Subscriber('/hefty_left/imu/data', Imu, angle_callback)

    # Spin the node to receive and process messages
    rospy.spin()