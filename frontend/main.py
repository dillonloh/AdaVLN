from flask import Flask, jsonify, send_from_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import base64
import threading
import json
import cv2
from cv_bridge import CvBridge
import numpy as np

app = Flask(__name__, static_folder='public')
PORT = 3000

# Data holders for latest images and decision data
rgb_image_base64 = None
depth_image_base64 = None
decision_data = {"decision": "N/A", "reasoning": "N/A"}

# Initialize ROS node and CvBridge
class FrontendServerNode(Node):
    def __init__(self):
        super().__init__('frontend_server_node')
        self.bridge = CvBridge()
        
        # Subscribe to the RGB image topic
        self.create_subscription(
            Image,
            '/rgb',
            self.rgb_callback,
            10
        )
        
        # Subscribe to the depth image topic
        self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )

        # Subscribe to the command reasoning topic
        self.create_subscription(
            String,
            '/command_reasoning',
            self.command_callback,
            10
        )

    def rgb_callback(self, msg):
        global rgb_image_base64
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Encode OpenCV image to JPEG
        _, buffer = cv2.imencode('.jpg', cv_image)
        # Convert JPEG image to base64
        rgb_image_base64 = base64.b64encode(buffer).decode('utf-8')
        self.get_logger().info(f"RGB Image Base64 length: {len(rgb_image_base64)}")

    def depth_callback(self, msg):
        global depth_image_base64
        # Convert ROS Image message to OpenCV image (assumed float32 depth data)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Set min and max depth values for visualization (similar to RViz2 settings)
        min_depth, max_depth = 0.0, 10.0
        cv_image_clipped = np.clip(cv_image, min_depth, max_depth)

        # Normalize to the 0-255 range for visualization as an 8-bit image
        cv_image_normalized = cv2.normalize(cv_image_clipped, None, 0, 255, cv2.NORM_MINMAX)
        cv_image_normalized = cv_image_normalized.astype('uint8')
        
        # Encode normalized depth image to JPEG
        _, buffer = cv2.imencode('.jpg', cv_image_normalized)
        depth_image_base64 = base64.b64encode(buffer).decode('utf-8')
        self.get_logger().info(f"Depth Image Base64 length: {len(depth_image_base64)}")


    def command_callback(self, msg):
        global decision_data
        try:
            parsed_command = json.loads(msg.data)
            decision_data = {
                "decision": parsed_command.get("decision", "N/A"),
                "reasoning": parsed_command.get("reasoning", "N/A")
            }
        except json.JSONDecodeError:
            self.get_logger().info("Received plain text command")
            decision_data = {
                "decision": msg.data,
                "reasoning": "No detailed reasoning provided"
            }

# ROS initialization
def init_ros_node():
    rclpy.init()
    node = FrontendServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Start ROS node in a separate thread
ros_thread = threading.Thread(target=init_ros_node, daemon=True)
ros_thread.start()

# API endpoint to get the latest RGB and Depth images and decision data
@app.route('/api/decision', methods=['GET'])
def get_decision_data():
    if rgb_image_base64 and depth_image_base64:
        return jsonify({
            "rgb_image": f"data:image/jpeg;base64,{rgb_image_base64}",
            "depth_image": f"data:image/jpeg;base64,{depth_image_base64}",
            "decision": decision_data["decision"],
            "reasoning": decision_data["reasoning"]
        })
    else:
        return jsonify({"error": "Images not available"}), 500

# Serve the HTML file for the frontend
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Start the server
if __name__ == '__main__':
    app.run(port=PORT)
