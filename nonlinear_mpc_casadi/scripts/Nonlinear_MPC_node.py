#!/usr/bin/env python3
from casadi import *
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import csv
import os
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from Nonlinear_MPC import MPC


class MPCKinematicNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        qos = QoSProfile(depth=10)

        # Parameters
        self.param = {
            'dT': self._declare_and_get('dT', 0.2),
            'N': self._declare_and_get('mpc_steps_N', 20),
            'L': self._declare_and_get('vehicle_L', 0.325),
            'theta_max': self._declare_and_get('mpc_max_steering', 0.523),
            'v_max': self._declare_and_get('max_speed', 2.0),
            'p_min': self._declare_and_get('p_min', 0.0),
            'p_max': self._declare_and_get('p_max', 3.0),
            'x_min': self._declare_and_get('x_min', -200.0),
            'x_max': self._declare_and_get('x_max', 200.0),
            'y_min': self._declare_and_get('y_min', -200.0),
            'y_max': self._declare_and_get('y_max', 200.0),
            'psi_min': self._declare_and_get('psi_min', -1000.0),
            'psi_max': self._declare_and_get('psi_max', 1000.0),
            's_min': self._declare_and_get('s_min', 0.0),
            's_max': self._declare_and_get('s_max', 200.0),
            'd_v_bound': self._declare_and_get('d_v_bound', 2.0),
            'd_theta_bound': self._declare_and_get('d_theta_bound', 0.5),
            'd_p_bound': self._declare_and_get('d_p_bound', 2.0),
            'ref_vel': self._declare_and_get('mpc_ref_vel', 2.0),
            'mpc_w_cte': self._declare_and_get('mpc_w_cte', 750.0),
            'mpc_w_s': self._declare_and_get('mpc_w_s', 0.0),
            'mpc_w_lag': self._declare_and_get('mpc_w_lag', 750.0),
            'mpc_w_vel': self._declare_and_get('mpc_w_vel', 0.75),
            'mpc_w_delta': self._declare_and_get('mpc_w_delta', 50.0),
            'mpc_w_p': self._declare_and_get('mpc_w_p', 5.0),
            'mpc_w_accel': self._declare_and_get('mpc_w_accel', 4.0),
            'mpc_w_delta_d': self._declare_and_get('mpc_w_delta_d', 750.0),
            'mpc_w_delta_p': self._declare_and_get('mpc_w_delta_p', 0.0),
            'spline_poly_order': self._declare_and_get('spline_poly_order', 3),
            'INTEGRATION_MODE': self._declare_and_get('integration_mode', 'Euler'),
            'ipopt_verbose': self._declare_and_get('ipopt_verbose', True),
            'ipopt_jit': self._declare_and_get('ipopt_jit', False),
            'ipopt_linear_solver': self._declare_and_get('ipopt_linear_solver', 'ma57')
        }

        path_folder_name = self._declare_and_get('path_folder_name', 'kelley')
        share_dir = get_package_share_directory('nonlinear_mpc_casadi')
        # Prefer installed layout: share/<pkg>/<path_folder_name>
        candidates = [
            os.path.join(share_dir, path_folder_name),
            os.path.join(share_dir, 'scripts', path_folder_name),
            os.path.join(os.path.dirname(__file__), path_folder_name),
        ]
        data_dir = next((p for p in candidates if os.path.isdir(p)), candidates[-1])
        self.CENTER_TRACK_FILENAME = os.path.join(data_dir, 'centerline_waypoints.csv')
        self.CENTER_DERIVATIVE_FILENAME = os.path.join(data_dir, 'center_spline_derivatives.csv')
        self.RIGHT_TRACK_FILENAME = os.path.join(data_dir, 'right_waypoints.csv')
        self.LEFT_TRACK_FILENAME = os.path.join(data_dir, 'left_waypoints.csv')
        self.CONTROLLER_FREQ = self._declare_and_get('controller_freq', 20)
        self.GOAL_THRESHOLD = self._declare_and_get('goal_threshold', 0.75)
        self.CAR_WIDTH = self._declare_and_get('car_width', 0.30)
        self.INFLATION_FACTOR = self._declare_and_get('inflation_factor', 0.9)
        self.LAG_TIME = self._declare_and_get('lag_time', 0.1)  # 100ms

        self.DEBUG_MODE = self._declare_and_get('debug_mode', True)
        self.DELAY_MODE = self._declare_and_get('delay_mode', True)
        self.THROTTLE_MODE = self._declare_and_get('throttle_mode', True)
        # Topic name related parameters
        pose_topic = self._declare_and_get('localized_pose_topic_name', '/pf/viz/inferred_pose')
        self.pose_topic_is_odom = self._declare_and_get('localized_pose_is_odom', False)
        cmd_vel_topic = self._declare_and_get('cmd_vel_topic_name', '/drive')
        odom_topic = self._declare_and_get('odom_topic_name', '/vesc/odom')
        goal_topic = self._declare_and_get('goal_topic_name', '/move_base_simple/goal')
        prediction_pub_topic = self._declare_and_get('mpc_prediction_topic', 'mpc_prediction')
        self.car_frame = self._declare_and_get('car_frame', 'base_link')

        # Path related variables
        self.path_points = None
        self.center_lane = None
        self.center_point_angles = None
        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None
        self.element_arc_lengths_orig = None

        # Plot related variables
        self.current_time = 0
        self.t_plot = []
        self.v_plot = []
        self.steering_plot = []
        self.cte_plot = []
        self.time_plot = []

        # Minimum distance search related variables
        self.ARC_LENGTH_MIN_DIST_TOL = self._declare_and_get('arc_length_min_dist_tol', 0.05)

        # Publishers
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, cmd_vel_topic, qos)
        self.mpc_trajectory_pub = self.create_publisher(Path, '/mpc_trajectory', qos)
        self.center_path_pub = self.create_publisher(Path, '/center_path', qos)
        self.right_path_pub = self.create_publisher(Path, '/right_path', qos)
        self.left_path_pub = self.create_publisher(Path, '/left_path', qos)
        self.center_tangent_pub = self.create_publisher(PoseStamped, '/center_tangent', qos)
        self.path_boundary_pub = self.create_publisher(MarkerArray, '/boundary_marker', qos)
        self.prediction_pub = self.create_publisher(MarkerArray, prediction_pub_topic, qos)

        # Pre-allocate markers for MPC predicted trajectory visualization.
        self.mpc_prediction_markers = self._init_marker_array(self.param['N'] + 1, color=(1.0, 0.0, 0.0))

        # MPC related initializations
        self.mpc = MPC()
        self.mpc.boundary_pub = self.path_boundary_pub
        self.mpc.logger = self.get_logger()
        self.initialize_MPC()
        self.current_pos_x, self.current_pos_y, self.current_yaw, self.current_s = 0.0, 0.0, 0.0, 0.0
        self.current_pose = None
        self.current_vel_odom = 0.0
        self.projected_vel = 0.0
        self.steering_angle = 0.0
        # Goal status related variables
        self.goal_pos = None
        self.goal_reached = False
        self.goal_received = False

        # Subscribers
        if self.pose_topic_is_odom:
            self.create_subscription(Odometry, pose_topic, self.odom_pose_callback, qos)
        else:
            self.create_subscription(PoseStamped, pose_topic, self.pf_pose_callback, qos)
        self.create_subscription(PoseStamped, goal_topic, self.goalCB, qos)
        self.create_subscription(Odometry, odom_topic, self.odomCB, qos)
        # Timer callback function for the control loop
        self.create_timer(1.0 / float(self.CONTROLLER_FREQ), self.controlLoopCB)

    def _declare_and_get(self, name, default_value):
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).value

    def initialize_MPC(self):
        self.preprocess_track_data()
        self.param['s_max'] = self.element_arc_lengths[-1]
        self.mpc.set_initial_params(self.param)
        self.mpc.set_track_data(self.center_lut_x, self.center_lut_y, self.center_lut_dx, self.center_lut_dy,
                                self.right_lut_x, self.right_lut_y, self.left_lut_x, self.left_lut_y,
                                self.element_arc_lengths, self.element_arc_lengths_orig[-1])
        self.mpc.setup_MPC()

    def create_header(self, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header

    def find_nearest_index(self, car_pos):
        distances_array = np.linalg.norm(self.center_lane - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        return min_dist_idx, distances_array[min_dist_idx]

    def heading(self, yaw):
        q = R.from_euler('z', yaw).as_quat()
        return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    def quaternion_to_euler_yaw(self, orientation):
        yaw = R.from_quat([
            orientation.x, orientation.y, orientation.z, orientation.w
        ]).as_euler('xyz')[2]
        return float(yaw)

    def _init_marker_array(self, num_markers, color=(1.0, 0.0, 1.0)):
        marker_array = MarkerArray()
        for i in range(num_markers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 0.5
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.2
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker_array.markers.append(marker)
        return marker_array

    def _update_marker_array(self, marker_array: MarkerArray, trajectory):
        num_update = min(len(marker_array.markers), trajectory.shape[0])
        now = self.get_clock().now().to_msg()
        for i in range(num_update):
            marker = marker_array.markers[i]
            marker.header.stamp = now
            marker.pose.position.x = float(trajectory[i, 0])
            marker.pose.position.y = float(trajectory[i, 1])
            if trajectory.shape[1] > 2:
                marker.pose.orientation = self.heading(float(trajectory[i, 2]))
        for i in range(num_update, len(marker_array.markers)):
            marker = marker_array.markers[i]
            marker.action = Marker.DELETE
        return marker_array

    def read_waypoints_array_from_csv(self, filename):
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = np.array([[float(point[0]), float(point[1])] for point in path_points])
        return path_points

    def pf_pose_callback(self, msg):
        self.current_pos_x = msg.pose.position.x
        self.current_pos_y = msg.pose.position.y
        self.current_yaw = self.quaternion_to_euler_yaw(msg.pose.orientation)
        self.current_pose = [self.current_pos_x, self.current_pos_y, self.current_yaw]
        if self.goal_received:
            car2goal_x = self.goal_pos.x - self.current_pos_x
            car2goal_y = self.goal_pos.y - self.current_pos_y
            dist2goal = sqrt(car2goal_x * car2goal_x + car2goal_y * car2goal_y)
            if dist2goal < self.GOAL_THRESHOLD:
                self.goal_reached = True
                self.goal_received = False
                self.mpc.WARM_START = False
                self.mpc.init_mpc_start_conditions()
                self.get_logger().info("Goal Reached!")
                self.plot_data()

    def odom_pose_callback(self, msg: Odometry):
        """Use odometry pose directly when no localization node is running."""
        self.current_pos_x = msg.pose.pose.position.x
        self.current_pos_y = msg.pose.pose.position.y
        self.current_yaw = self.quaternion_to_euler_yaw(msg.pose.pose.orientation)
        self.current_pose = [self.current_pos_x, self.current_pos_y, self.current_yaw]

    def odomCB(self, msg):
        self.current_vel_odom = msg.twist.twist.linear.x

    def goalCB(self, msg):
        self.goal_pos = msg.pose.position
        self.goal_received = True
        self.goal_reached = False
        if self.DEBUG_MODE:
            print("Goal pos=", self.goal_pos)

    def publish_path(self, waypoints, publisher):
        path = Path()
        path.header = self.create_header('map')
        path.poses = []
        for point in waypoints:
            tempPose = PoseStamped()
            tempPose.header = path.header
            tempPose.pose.position.x = point[0]
            tempPose.pose.position.y = point[1]
            tempPose.pose.orientation.w = 1.0
            path.poses.append(tempPose)
        publisher.publish(path)

    def get_interpolated_path(self, pts, arc_lengths_arr, smooth_value=0.1, scale=2, derivative_order=0):
        tck, u = splprep(pts.T, u=arc_lengths_arr, s=smooth_value, per=1)
        u_new = np.linspace(u.min(), u.max(), len(pts) * scale)
        x_new, y_new = splev(u_new, tck, der=derivative_order)
        interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
        return interp_points, tck

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = interpolant(label_x, 'bspline', [u], V_X)
        lut_y = interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x, lut_y

    def get_arc_lengths(self, waypoints):
        d = np.diff(waypoints, axis=0)
        consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
        dists_cum = np.cumsum(consecutive_diff)
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum

    def inflate_track_boundaries(self, center_lane, side_lane, car_width=0.325, inflation_factor=1.2):
        for idx in range(len(center_lane)):
            lane_vector = side_lane[idx, :] - center_lane[idx, :]
            side_track_width = np.linalg.norm(lane_vector)
            side_unit_vector = lane_vector / side_track_width
            side_lane[idx, :] = center_lane[idx, :] + side_unit_vector * (
                    side_track_width - car_width * inflation_factor)
        return side_lane

    def preprocess_track_data(self):
        center_lane = self.read_waypoints_array_from_csv(self.CENTER_TRACK_FILENAME)
        center_derivative_data = self.read_waypoints_array_from_csv(self.CENTER_DERIVATIVE_FILENAME)
        right_lane = self.read_waypoints_array_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_array_from_csv(self.LEFT_TRACK_FILENAME)

        for _ in range(5):
            self.publish_path(center_lane, self.center_path_pub)
            self.publish_path(right_lane, self.right_path_pub)
            self.publish_path(left_lane, self.left_path_pub)
            time.sleep(0.2)

        right_lane = self.inflate_track_boundaries(center_lane, right_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)
        left_lane = self.inflate_track_boundaries(center_lane, left_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)

        self.center_lane = np.row_stack((center_lane, center_lane[1:int(center_lane.shape[0] / 2), :]))
        right_lane = np.row_stack((right_lane, right_lane[1:int(center_lane.shape[0] / 2), :]))
        left_lane = np.row_stack((left_lane, left_lane[1:int(center_lane.shape[0] / 2), :]))
        center_derivative_data = np.row_stack(
            (center_derivative_data, center_derivative_data[1:int(center_lane.shape[0] / 2), :]))

        self.element_arc_lengths_orig = self.get_arc_lengths(center_lane)
        self.element_arc_lengths = self.get_arc_lengths(self.center_lane)
        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y',
                                                                                 self.center_lane,
                                                                                 self.element_arc_lengths)
        self.center_lut_dx, self.center_lut_dy = self.get_interpolated_path_casadi('lut_center_dx', 'lut_center_dy',
                                                                                   center_derivative_data,
                                                                                   self.element_arc_lengths)
        self.center_point_angles = np.arctan2(center_derivative_data[:, 1], center_derivative_data[:, 0])

        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_lane,
                                                                               self.element_arc_lengths)
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_lane,
                                                                             self.element_arc_lengths)

    def find_current_arc_length(self, car_pos):
        nearest_index, minimum_dist = self.find_nearest_index(car_pos)
        if minimum_dist > self.ARC_LENGTH_MIN_DIST_TOL:
            if nearest_index == 0:
                next_idx = 1
                prev_idx = self.center_lane.shape[0] - 1
            elif nearest_index == (self.center_lane.shape[0] - 1):
                next_idx = 0
                prev_idx = self.center_lane.shape[0] - 2
            else:
                next_idx = nearest_index + 1
                prev_idx = nearest_index - 1
            dot_product_value = np.dot(car_pos - self.center_lane[nearest_index, :],
                                       self.center_lane[prev_idx, :] - self.center_lane[nearest_index, :])
            if dot_product_value > 0:
                nearest_index_actual = prev_idx
            else:
                nearest_index_actual = nearest_index
                nearest_index = next_idx
            new_dot_value = np.dot(car_pos - self.center_lane[nearest_index_actual, :],
                                   self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            projection = new_dot_value / np.linalg.norm(
                self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            current_s = self.element_arc_lengths[nearest_index_actual] + projection
        else:
            current_s = self.element_arc_lengths[nearest_index]

        if nearest_index == 0:
            current_s = 0.0
        return current_s, nearest_index

    def controlLoopCB(self):
        if self.goal_received and not self.goal_reached:
            control_loop_start_time = time.time()
            px = self.current_pos_x
            py = self.current_pos_y
            car_pos = np.array([self.current_pos_x, self.current_pos_y])
            psi = self.current_yaw

            v = self.current_vel_odom
            steering = self.steering_angle
            L = self.mpc.L

            current_s, near_idx = self.find_current_arc_length(car_pos)
            print("pre", current_s, near_idx)
            if self.DELAY_MODE:
                dt_lag = self.LAG_TIME
                px = px + v * np.cos(psi) * dt_lag
                py = py + v * np.sin(psi) * dt_lag
                psi = psi + (v / L) * tan(steering) * dt_lag
                current_s = current_s + self.projected_vel * dt_lag

            current_state = np.array([px, py, psi, current_s])

            centerPose = PoseStamped()
            centerPose.header = self.create_header('map')
            centerPose.pose.position.x = float(self.center_lane[near_idx, 0])
            centerPose.pose.position.y = float(self.center_lane[near_idx, 1])
            centerPose.pose.orientation = self.heading(self.center_point_angles[near_idx])
            self.center_tangent_pub.publish(centerPose)

            mpc_time = time.time()
            first_control, trajectory, control_inputs = self.mpc.solve(current_state)
            mpc_compute_time = time.time() - mpc_time

            speed = float(first_control[0])
            steering = float(first_control[1])
            self.projected_vel = speed

            throttle = 0.03 * (speed - v) / self.param['dT']

            if throttle > 1:
                throttle = 1
            elif throttle < -1:
                throttle = -1
            if speed == 0:
                throttle = 0

            if not self.mpc.WARM_START:
                speed, steering, throttle = 0, 0, 0
                self.mpc.WARM_START = True
            if speed >= self.param['v_max']:
                speed = self.param['v_max']
            elif speed <= (- self.param['v_max'] / 2.0):
                speed = - self.param['v_max'] / 2.0

            # Ensure Python floats for ROS message compatibility.
            speed = float(speed)
            steering = float(steering)
            throttle = float(throttle)

            mpc_traj = Path()
            mpc_traj.header = self.create_header('map')
            mpc_traj.poses = []
            for i in range(trajectory.shape[0]):
                tempPose = PoseStamped()
                tempPose.header = mpc_traj.header
                tempPose.pose.position.x = trajectory[i, 0]
                tempPose.pose.position.y = trajectory[i, 1]
                tempPose.pose.orientation = self.heading(trajectory[i, 2])
                mpc_traj.poses.append(tempPose)
            self.mpc_trajectory_pub.publish(mpc_traj)
            if self.prediction_pub.get_subscription_count() > 0:
                self.prediction_pub.publish(
                    self._update_marker_array(self.mpc_prediction_markers, trajectory)
                )

            total_time = time.time() - control_loop_start_time
            if self.DEBUG_MODE:
                self.get_logger().info(f"psi: {psi}")
                self.get_logger().info(f"V: {v}")
                self.get_logger().info(f"Throttle: {throttle}")
                self.get_logger().info(f"Control loop time mpc= {mpc_compute_time}")
                self.get_logger().info(f"Control loop time= {total_time}")

            self.current_time += 1.0 / self.CONTROLLER_FREQ
            self.t_plot.append(self.current_time)
            self.v_plot.append(speed)
            self.steering_plot.append(np.rad2deg(steering))
            self.time_plot.append(mpc_compute_time * 1000)
        else:
            steering = 0.0
            speed = 0.0
            throttle = 0.0

        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = float(steering)
        self.steering_angle = steering
        ackermann_cmd.drive.speed = float(speed)
        if self.THROTTLE_MODE:
            ackermann_cmd.drive.acceleration = float(throttle)
        self.ackermann_pub.publish(ackermann_cmd)

    def plot_data(self):
        plt.figure(1)
        plt.subplot(411)
        plt.step(self.t_plot, self.v_plot, 'k', linewidth=1.5)
        plt.ylabel('v m/s')
        plt.xlabel('time(s)')
        plt.subplot(412)
        plt.step(self.t_plot, self.steering_plot, 'r', linewidth=1.5)
        plt.ylabel('steering angle(degrees)')
        plt.xlabel('time(s)')
        plt.subplot(414)
        plt.step(self.t_plot, self.time_plot, 'b', linewidth=1.5)
        plt.ylim(0.0, 100)
        plt.ylabel('mpc_compute_time in ms')
        plt.xlabel('time(s)')
        plt.show()

        self.t_plot = []
        self.steering_plot = []
        self.v_plot = []
        self.cte_plot = []
        self.time_plot = []
        self.current_time = 0


def main(args=None):
    rclpy.init(args=args)
    node = MPCKinematicNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
