from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('nonlinear_mpc_casadi')
    params_file = os.path.join(pkg_share, 'params', 'mpcc_params_f1tenth_gym.yaml.yaml')

    return LaunchDescription([
        Node(
            package='nonlinear_mpc_casadi',
            executable='Nonlinear_MPC_node.py',
            name='mpc_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
