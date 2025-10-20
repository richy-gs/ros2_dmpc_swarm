from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = FindPackageShare('mpc_drone_viz')
    
    # Declare the RViz configuration file parameter
    rviz_config_file = PathJoinSubstitution(
        [pkg_share, 'rviz', 'mpc_drone.rviz'])

    return LaunchDescription([
        # Launch the MPC controller node
        Node(
            package='mpc_drone_viz',
            executable='mpc_controller_ca',
            # name='mpc_drone_controller',
            output='screen'
        ),
        
        # Launch RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )
    ])