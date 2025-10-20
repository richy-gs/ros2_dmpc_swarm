#!/usr/bin/env python3
"""
Approach 1: Multi-Drone MPC with Independent Controllers
Each drone solves MPC independently. No inter-drone collision avoidance.
Drones can visualize each other but don't coordinate.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
import casadi as ca


class MultiDroneMPCController(Node):
    """
    Multi-drone MPC controller where each drone is independent.
    Drones share visualization but no control coordination.
    """
    
    def __init__(self, drone_id=0, num_drones=3):
        super().__init__(f'mpc_drone_controller_{drone_id}')
        
        self.drone_id = drone_id
        self.num_drones = num_drones
        
        # Setup static transform broadcaster for world frame (only from drone 0)
        if drone_id == 0:
            self.tf_broadcaster = StaticTransformBroadcaster(self)
            static_transform = TransformStamped()
            static_transform.header.stamp = self.get_clock().now().to_msg()
            static_transform.header.frame_id = 'map'
            static_transform.child_frame_id = 'world'
            static_transform.transform.translation.x = 0.0
            static_transform.transform.translation.y = 0.0
            static_transform.transform.translation.z = 0.0
            static_transform.transform.rotation.x = 0.0
            static_transform.transform.rotation.y = 0.0
            static_transform.transform.rotation.z = 0.0
            static_transform.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(static_transform)
        
        # Publishers for this drone
        self.drone_pose_pub = self.create_publisher(
            PoseStamped, f'/drone_{drone_id}/pose', 10)
        self.trajectory_pub = self.create_publisher(
            MarkerArray, f'/drone_{drone_id}/mpc_trajectory', 10)
        
        # Shared obstacle publisher (only from drone 0)
        if drone_id == 0:
            self.obstacle_pub = self.create_publisher(
                MarkerArray, '/environment/obstacles', 10)
        
        # Publishers for other drones' positions (for visualization)
        self.other_drones_pub = self.create_publisher(
            MarkerArray, f'/drone_{drone_id}/other_drones', 10)
        
        # Subscribers to other drones' poses
        self.drone_positions = {}
        for i in range(num_drones):
            if i != drone_id:
                self.create_subscription(
                    PoseStamped,
                    f'/drone_{i}/pose',
                    self.create_pose_callback(i),
                    10)
        
        # MPC Parameters
        self.N = 15  # Prediction horizon
        self.dt = 0.1  # Time step
        self.max_vel = 0.3  # Max velocity (m/s)
        self.max_acc = 0.1  # Max acceleration (m/s^2)
        
        # Drone-specific starting positions (spread out in a line)
        start_x = drone_id * 1.5
        self.drone_state = np.array([start_x, 0.0, 0.5, 0.0, 0.0, 0.0])
        
        # Shared target for all drones
        self.target = np.array([5.0, 5.0, 2.0])
        
        # Shared obstacles
        self.obstacles = [
            {'center': np.array([1.3, 1.3, 1.0]), 'radius': 0.5},
            {'center': np.array([2.9, 2.5, 1.5]), 'radius': 0.6},
            {'center': np.array([4.3, 3.7, 2.0]), 'radius': 0.7},
            {'center': np.array([4.0, 3.0, 2.0]), 'radius': 1.0},
        ]
        
        # Setup MPC solver
        self.setup_mpc()
        
        # Control loop
        self.timer = self.create_timer(0.1, self.control_step)
        self.get_logger().info(f'Drone {drone_id} initialized')
    
    def create_pose_callback(self, drone_id):
        """Factory function to create callbacks for each drone's pose"""
        def callback(msg):
            self.drone_positions[drone_id] = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
        return callback
    
    def setup_mpc(self):
        """Setup CasADi MPC solver"""
        opti = ca.Opti()
        
        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)
        X0 = opti.parameter(6)
        opti.subject_to(X[:, 0] == X0)
        
        # Dynamics constraints
        for k in range(self.N):
            opti.subject_to(X[0:3, k + 1] == X[0:3, k] + X[3:6, k] * self.dt)
            opti.subject_to(X[3:6, k + 1] == X[3:6, k] + U[:, k] * self.dt)
        
        # Input and velocity constraints
        opti.subject_to(opti.bounded(-self.max_acc, U, self.max_acc))
        opti.subject_to(opti.bounded(-self.max_vel, X[3:6, :], self.max_vel))
        
        # Minimum altitude constraint (drones don't touch ground)
        opti.subject_to(X[2, :] >= 0.0)  # z >= 0.0 for all time steps
        
        # Cost function
        target_cost = 10.0 * ca.sumsqr(X[0:3, -1] - self.target)
        effort_cost = 0.1 * ca.sumsqr(U)
        
        # Soft obstacle avoidance
        obstacle_cost = 0
        for obs in self.obstacles:
            center = obs['center']
            radius = obs['radius']
            safety_margin = 1.0
            
            for k in range(self.N + 1):
                pos = X[0:3, k]
                dist = ca.sqrt((pos[0] - center[0])**2 + 
                               (pos[1] - center[1])**2 + 
                               (pos[2] - center[2])**2)
                min_dist = radius + safety_margin
                penalty = ca.fmax(0, min_dist - dist)**2
                obstacle_cost += 25.0 * penalty
        
        opti.minimize(target_cost + effort_cost + obstacle_cost)
        
        opti.solver('ipopt', {
            'print_time': False,
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-4
            }
        })
        
        self.opti = opti
        self.X = X
        self.U = U
        self.X0 = X0
    
    def control_step(self):
        """Execute MPC control step"""
        try:
            self.opti.set_value(self.X0, self.drone_state)
            sol = self.opti.solve()
            
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)
            
            u_opt = U_opt[:, 0]
            self.drone_state[3:6] += u_opt * self.dt
            
            vel_norm = np.linalg.norm(self.drone_state[3:6])
            if vel_norm > self.max_vel:
                self.drone_state[3:6] *= self.max_vel / vel_norm
            
            self.drone_state[0:3] += self.drone_state[3:6] * self.dt
            
            # Publish drone pose
            self.publish_drone_pose()
            
            # Publish MPC trajectory
            self.publish_trajectory(X_opt)
            
            # Publish visualization of other drones
            self.publish_other_drones()
            
            # Publish obstacles (only from drone 0)
            if self.drone_id == 0:
                self.publish_obstacles()
            
        except Exception as e:
            self.get_logger().warn(f'Drone {self.drone_id} solve failed: {str(e)[:100]}')
    
    def publish_drone_pose(self):
        """Publish this drone's pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(self.drone_state[0])
        pose_msg.pose.position.y = float(self.drone_state[1])
        pose_msg.pose.position.z = float(self.drone_state[2])
        self.drone_pose_pub.publish(pose_msg)
    
    def publish_trajectory(self, X_opt):
        """Publish predicted trajectory as markers"""
        markers = MarkerArray()
        
        # Color for this drone (different for each)
        colors = [
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow
        ]
        color = colors[self.drone_id % len(colors)]
        
        # Trajectory line
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = 'world'
        line_marker.id = self.drone_id * 1000
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color = color
        
        for k in range(X_opt.shape[1]):
            pt = Point()
            pt.x = float(X_opt[0, k])
            pt.y = float(X_opt[1, k])
            pt.z = float(X_opt[2, k])
            line_marker.points.append(pt)
        
        markers.markers.append(line_marker)
        
        # Predicted poses
        for k in range(0, X_opt.shape[1], 2):
            sphere_marker = Marker()
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.header.frame_id = 'world'
            sphere_marker.id = self.drone_id * 1000 + k + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(X_opt[0, k])
            sphere_marker.pose.position.y = float(X_opt[1, k])
            sphere_marker.pose.position.z = float(X_opt[2, k])
            sphere_marker.scale.x = 0.05
            sphere_marker.scale.y = 0.05
            sphere_marker.scale.z = 0.05
            sphere_marker.color = color
            markers.markers.append(sphere_marker)
        
        # Target (shared, publish from all drones)
        target_marker = Marker()
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.header.frame_id = 'world'
        target_marker.id = 9999
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = float(self.target[0])
        target_marker.pose.position.y = float(self.target[1])
        target_marker.pose.position.z = float(self.target[2])
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
        markers.markers.append(target_marker)
        
        self.trajectory_pub.publish(markers)
    
    def publish_other_drones(self):
        """Visualize other drones' current positions"""
        markers = MarkerArray()
        
        for drone_id, pos in self.drone_positions.items():
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'world'
            marker.id = drone_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            
            # Color coding for other drones
            if drone_id == 0:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            elif drone_id == 1:
                marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
            else:
                marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
            
            markers.markers.append(marker)
        
        self.other_drones_pub.publish(markers)
    
    def publish_obstacles(self):
        """Publish obstacles (from drone 0 only)"""
        markers = MarkerArray()
        
        for i, obs in enumerate(self.obstacles):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'world'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obs['center'][0])
            marker.pose.position.y = float(obs['center'][1])
            marker.pose.position.z = float(obs['center'][2])
            marker.scale.x = obs['radius'] * 2
            marker.scale.y = obs['radius'] * 2
            marker.scale.z = obs['radius'] * 2
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            markers.markers.append(marker)
        
        self.obstacle_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    
    # Create 3 drone controllers
    num_drones = 3
    nodes = []
    for i in range(num_drones):
        node = MultiDroneMPCController(drone_id=i, num_drones=num_drones)
        nodes.append(node)
    
    # Spin all nodes
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    
    executor.spin()
    
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()