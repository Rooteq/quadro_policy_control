from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import (
    LaunchConfiguration,
    IfElseSubstitution,
    TextSubstitution,
)
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('quadro_policy_control'),
        'policy/policy.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "publish_period_ms",
            default_value="5",
            description="publishing dt in milliseconds"),
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the policy file"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        DeclareLaunchArgument(
            "namespace",
            default_value="h1_01",
            description="ROS namespace for the H1 controller"),
        DeclareLaunchArgument(
            "use_namespace",
            default_value="False",
            description="Whether to apply the ROS namespace to the node"),
        Node(
            package='quadro_policy_control',
            executable='quadro_controller',
            name='quadro_policy_controller',
            output="screen",
            namespace=IfElseSubstitution(
                [LaunchConfiguration('use_namespace')],
                [LaunchConfiguration('namespace')],
                [TextSubstitution(text='')]
            ),
            parameters=[{
                'publish_period_ms': LaunchConfiguration('publish_period_ms'),
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
            }]

        ),
    ])