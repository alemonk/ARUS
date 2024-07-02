#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import os
import re
import cv2
from my_package.msg import ImagePose
import random

randomize_images = False

def get_pose(i):
    distance_cm = 0.045

    # Initial pose
    initial_x = 0
    initial_y = 0
    initial_z = 0
    qw = 0.7071
    qx = 0
    qy = 0.7071
    qz = 0

    # Calculate the new x-coordinate for the image
    x = initial_x + i * distance_cm
    y = initial_y
    z = initial_z

    # Return the pose as a Pose object
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    return pose

def collect_images(image_dir):
    bridge = CvBridge()
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)
    
    images_with_poses = []

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if cv_image is not None:
            ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
            title = image_file
            pose = get_pose(i)
            
            image_pose_msg = ImagePose()
            image_pose_msg.image = ros_image
            image_pose_msg.title = title
            image_pose_msg.pose = pose

            images_with_poses.append(image_pose_msg)
        else:
            rospy.logwarn(f"Failed to read {image_path}")

    return images_with_poses

def publish_images_randomized(images_with_poses, pub):
    rate = rospy.Rate(2) # Hz

    if randomize_images:
        random.shuffle(images_with_poses)

    for image_pose_msg in images_with_poses:
        pub.publish(image_pose_msg)
        rospy.loginfo(f"Published {image_pose_msg.title}")
        rate.sleep()

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def main():
    rospy.init_node('image_publisher_node')
    default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
    image_dir = rospy.get_param('~image_dir', f'{default_path}test_forearm')
    image_topic = rospy.get_param('~image_topic', 'image_topic')
    
    pub = rospy.Publisher(image_topic, ImagePose, queue_size=100)
    rospy.loginfo(f"Starting to publish images from {image_dir} to {image_topic}")

    try:
        images_with_poses = collect_images(image_dir)
        publish_images_randomized(images_with_poses, pub)
        rospy.spin()  # Keeps the node running until shutdown
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
