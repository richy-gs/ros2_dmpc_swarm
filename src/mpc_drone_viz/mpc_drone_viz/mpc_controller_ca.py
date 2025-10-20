
#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
import casadi as ca

class MPCDroneController(Node):
    def __init__(self):
        super().__init__('mpc_drone_controller')
        
        # Setup static transform broadcaster for world frame
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
        
        # Publisher for drone pose and trajectory visualization
        self.drone_pose_pub = self.create_publisher(PoseStamped, '/drone/pose', 10)
        self.trajectory_pub = self.create_publisher(MarkerArray, '/drone/mpc_trajectory', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, '/drone/obstacles', 10)
        
        # MPC Parameters
        self.N = 10  # Prediction horizon
        self.dt = 0.1  # Time step
        self.max_vel = 0.3  # Max velocity (m/s)
        self.max_acc = 0.1  # Max acceleration (m/s^2)
        
        # State: [x, y, z, vx, vy, vz]
        self.drone_state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        self.target = np.array([5.0, 5.0, 2.0])
        self.obstacles = [
            # {'center': np.array([2.5, 2.5, 1.5]), 'radius': 0.4},
            {'center': np.array([1.3, 1.3, 1.0]), 'radius': 0.8},
            {'center': np.array([2.9, 2.5, 1.5]), 'radius': 0.6},
            {'center': np.array([3.0, 3.5, 2.0]), 'radius': 0.9},
            {'center': np.array([4.0, 3.0, 2.0]), 'radius': 0.6},
            {'center': np.array([4.3, 3.7, 2.0]), 'radius': 0.67},
            {'center': np.array([4.0, 3.0, 2.0]), 'radius': 1.0},
        ]
        
        # Setup MPC solver (simplified, no hard constraints)
        self.setup_mpc()
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_step)
        self.get_logger().info('MPC Drone Controller started')
    
    def setup_mpc(self):
        """Setup simplified MPC solver without hard obstacle constraints"""
        # Create optimization problem
        opti = ca.Opti()
        
        # Decision variables: position and velocity at each time step
        # State: [x, y, z, vx, vy, vz] for each time step
        X = opti.variable(6, self.N + 1)
        
        # Control input: acceleration at each time step [ax, ay, az]
        U = opti.variable(3, self.N)
        
        # Parameter for initial state
        X0 = opti.parameter(6)
        opti.subject_to(X[:, 0] == X0)
        
        # Dynamics constraints using simple Euler integration
        for k in range(self.N):
            # Position update: x_{k+1} = x_k + v_k * dt
            opti.subject_to(X[0:3, k + 1] == X[0:3, k] + X[3:6, k] * self.dt)
            
            # Velocity update: v_{k+1} = v_k + a_k * dt
            opti.subject_to(X[3:6, k + 1] == X[3:6, k] + U[:, k] * self.dt)
        
        # Input constraints (acceleration limits)
        opti.subject_to(opti.bounded(-self.max_acc, U, self.max_acc))
        
        # Velocity constraints
        opti.subject_to(opti.bounded(-self.max_vel, X[3:6, :], self.max_vel))
        
        # Cost function: track target position + minimize acceleration + avoid obstacles
        # The MPC tries to reach the target while minimizing energy and avoiding obstacles
        target_cost = 10.0 * ca.sumsqr(X[0:3, -1] - self.target)  # Weight final position heavily
        effort_cost = 0.1 * ca.sumsqr(U)  # Penalize acceleration to keep motion smooth
        
        # Obstacle avoidance cost: penalize being too close to obstacles
        # Instead of hard constraints that break the solver, we add a soft penalty
        obstacle_cost = 0
        for obs in self.obstacles:
            center = obs['center']
            radius = obs['radius']
            safety_margin = 1.0  # Desired minimum distance
            
            # For each predicted state, calculate distance to obstacle
            for k in range(self.N + 1):
                pos = X[0:3, k]
                # Distance from drone to obstacle center
                dist = ca.sqrt((pos[0] - center[0])**2 + 
                               (pos[1] - center[1])**2 + 
                               (pos[2] - center[2])**2)
                
                # If closer than safety margin, add penalty that grows as we get closer
                # This creates a "repulsive" effect that steers the drone away
                min_dist = radius + safety_margin
                # Penalty is high if dist < min_dist, zero otherwise
                penalty = ca.fmax(0, min_dist - dist)**2
                obstacle_cost += 25.0 * penalty
        
        opti.minimize(target_cost + effort_cost + obstacle_cost)
        
        # Solver options: keep it simple and fast
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
            # Update the initial state parameter for this iteration
            self.opti.set_value(self.X0, self.drone_state)
            
            # Solve MPC problem
            sol = self.opti.solve()
            
            # Extract optimal trajectory and controls
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)
            
            # Apply first control input to update drone state
            u_opt = U_opt[:, 0]
            
            # Update velocity: v = v + a*dt
            self.drone_state[3:6] += u_opt * self.dt
            
            # Clamp velocity to limits
            vel_norm = np.linalg.norm(self.drone_state[3:6])
            if vel_norm > self.max_vel:
                self.drone_state[3:6] *= self.max_vel / vel_norm
            
            # Update position: p = p + v*dt
            self.drone_state[0:3] += self.drone_state[3:6] * self.dt
            
            # Publish drone pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = float(self.drone_state[0])
            pose_msg.pose.position.y = float(self.drone_state[1])
            pose_msg.pose.position.z = float(self.drone_state[2])
            self.drone_pose_pub.publish(pose_msg)
            
            # Publish MPC trajectory visualization
            self.publish_trajectory(X_opt)
            
            # Publish obstacles
            self.publish_obstacles()
            
            self.get_logger().info(f'Drone at: {self.drone_state[0:3]}, Target: {self.target}')
            
        except Exception as e:
            self.get_logger().warn(f'MPC solve failed: {str(e)[:100]}')
    
    def publish_trajectory(self, X_opt):
        """Publish predicted trajectory as markers"""
        markers = MarkerArray()
        
        # Trajectory line (connects all predicted states)
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = 'world'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        # Add all predicted positions to the line
        for k in range(X_opt.shape[1]):
            pt = Point()
            pt.x = float(X_opt[0, k])
            pt.y = float(X_opt[1, k])
            pt.z = float(X_opt[2, k])
            line_marker.points.append(pt)
        
        markers.markers.append(line_marker)
        
        # Add spheres at intermediate points along trajectory
        for k in range(0, X_opt.shape[1], 2):
            sphere_marker = Marker()
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.header.frame_id = 'world'
            sphere_marker.id = k + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(X_opt[0, k])
            sphere_marker.pose.position.y = float(X_opt[1, k])
            sphere_marker.pose.position.z = float(X_opt[2, k])
            sphere_marker.scale.z = 0.05
            sphere_marker.scale.x = 0.05
            sphere_marker.scale.y = 0.05
            sphere_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            markers.markers.append(sphere_marker)
        
        # Target marker (large red sphere)
        target_marker = Marker()
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.header.frame_id = 'world'
        target_marker.id = 1000
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
    
    def publish_obstacles(self):
        """Publish obstacles as markers"""
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
            # Display obstacle with radius
            marker.scale.x = obs['radius'] * 2
            marker.scale.y = obs['radius'] * 2
            marker.scale.z = obs['radius'] * 2
            # Red semi-transparent color
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            markers.markers.append(marker)
        
        self.obstacle_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = MPCDroneController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()