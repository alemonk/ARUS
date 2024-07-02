#!/usr/bin/env python3

import rospy
from PIL import Image as PILImage
import torch
from torchvision import transforms
import re
import time
import threading
import queue
from unet import UNet
from cv_bridge import CvBridge
from my_package.msg import ImagePose
import numpy as np
from helper_functions import resize_image
from params import *

def run_model(model_class, model_path, image):
    # Instantiate model, loss function, and optimizer
    model = model_class(n_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load and preprocess image
    image = resize_image(image, img_height)
    image_tensor = final_transform(image).unsqueeze(0)

    colors = get_colors(n_class)

    with torch.no_grad():
        output = model(image_tensor.float())
        output = torch.sigmoid(output)
        output = output[0].cpu().numpy()

        # Apply threshold and color mapping
        color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
        for k in range(output.shape[0]):
            color_mask[output[k] > threshold] = colors[k % len(colors)]

    return color_mask

class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node('image_segmentation_node')
        default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
        self.model_path = rospy.get_param('~model_path', f'{default_path}best_model.model')
        self.image_topic = rospy.get_param('~image_topic', 'image_topic')
        self.processed_image_topic = rospy.get_param('~processed_image_topic', 'processed_image_topic')

        self.bridge = CvBridge()
        self.image_queue = queue.Queue()
        self.subscriber = rospy.Subscriber(self.image_topic, ImagePose, self.callback)
        self.publisher = rospy.Publisher(self.processed_image_topic, ImagePose, queue_size=100)
        
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def callback(self, image_pose_msg):
        self.image_queue.put(image_pose_msg)

    def process_images(self):
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                image_pose_msg = self.image_queue.get()
                image = image_pose_msg.image
                pose = image_pose_msg.pose
                title = image_pose_msg.title

                try:
                    cv_image = self.bridge.imgmsg_to_cv2(image, "mono8")
                    pil_image = PILImage.fromarray(cv_image)
                    processed_image = run_model(UNet, self.model_path, pil_image)

                    # Convert processed image back to ROS image message
                    processed_ros_image = self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8")

                    # Publish the processed image along with the pose and title
                    processed_image_pose_msg = ImagePose()
                    processed_image_pose_msg.image = processed_ros_image
                    processed_image_pose_msg.pose = pose
                    processed_image_pose_msg.title = title
                    self.publisher.publish(processed_image_pose_msg)

                    rospy.loginfo(f"Segmentation completed on image {title}")

                except Exception as e:
                    rospy.logerr(f"Error processing image: {e}")
            else:
                time.sleep(0.1)

def main():
    node = ImageSegmentationNode()
    rospy.spin()

if __name__ == '__main__':
    main()
