import rclpy
import torch
import numpy as np
import io
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
# from message_filters import Subscriber, ApproximateTimeSynchronizer


class QuadroController(Node):

    def __init__(self):
        super().__init__('quadro_controller')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/policy.pt')
        self.set_parameters(
            [rclpy.parameter.Parameter(
                'use_sim_time', 
                rclpy.Parameter.Type.BOOL, 
                True
            )]
        )

        self._logger = self.get_logger()
        
        # Configure QoS profile for simulation
        # sim_qos_profile = rclpy.qos.
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            # history=rclpy.qos.HistoryPolicy.KEEP_ALL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscription for velocity commands
        self._cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self._cmd_vel_callback,
            qos_profile=10)

        # Create publisher for joint commands
        self._joint_publisher = self.create_publisher(
            JointState,
            'joint_command',
            qos_profile=sim_qos_profile)

        # Create regular subscribers (not using message_filters)
        self._imu_subscription = self.create_subscription(
            Imu,
            'imu',
            self._imu_callback,
            qos_profile=sim_qos_profile)
        
        self._joint_states_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self._joint_states_callback,
            qos_profile=sim_qos_profile)
        
        self._logger.info("Configured subscribers, waiting for messages...")

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._action = np.zeros(12)
        self._action_scale = 0.2  # Scale factor for policy output
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._decimation = 4  # Run policy every 4 ticks to reduce computation
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks
        
        # Joint names in the order the policy expects (obs input / action output)
        self.policy_joint_names = [
            'bl_m1_s1',
            'br_m1_s1',
            'fl_m1_s1',
            'fr_m1_s1',
            'bl_m2_s2',
            'br_m2_s2',
            'fl_m2_s2',
            'fr_m2_s2',
            'bl_l4_l3',
            'br_l4_l3',
            'fl_l4_l3',
            'fr_l4_l3',
        ]

        # Joint names in the order the robot controller expects (ROS topics)
        self.robot_joint_names = [
            'bl_m1_s1',
            'br_m1_s1',
            'fl_m1_s1',
            'fr_m1_s1',
            'bl_m2_s2',
            'br_m2_s2',
            'fl_m2_s2',
            'fr_m2_s2',
            'bl_l4_l3',
            'br_l4_l3',
            'fl_l4_l3',
            'fr_l4_l3',
        ]

        # Mapping indices: policy order -> robot order and vice versa
        self._policy_to_robot = np.array([self.robot_joint_names.index(n) for n in self.policy_joint_names])
        self._robot_to_policy = np.array([self.policy_joint_names.index(n) for n in self.robot_joint_names])

        # Default joint positions in POLICY order
        # self.default_pos = np.array([
        #      0.785,  # fl_m1_s1
        #      0.436,  # fl_m2_s2
        #     -0.785,  # fl_l4_l3
        #     -0.785,  # fr_m1_s1
        #     -0.436,  # fr_m2_s2
        #      0.785,  # fr_l4_l3
        #     -0.785,  # bl_m1_s1
        #      0.436,  # bl_m2_s2
        #     -0.785,  # bl_l4_l3
        #      0.785,  # br_m1_s1
        #     -0.436,  # br_m2_s2
        #      0.785,  # br_l4_l3
        # ])

        self.default_pos = np.array([
             0.0,  # fl_m1_s1
             0.0,  # fl_m2_s2
            0.0,  # fl_l4_l3
            0.0,  # fr_m1_s1
            0.0,  # fr_m2_s2
             0.0,  # fr_l4_l3
            0.0,  # bl_m1_s1
             0.0,  # bl_m2_s2
            0.0,  # bl_l4_l3
             0.0,  # br_m1_s1
            0.0,  # br_m2_s2
             0.0,  # br_l4_l3
        ])

        self._logger.info("Initializing QuadroController")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _imu_callback(self, msg: Imu):
        """Store the latest IMU data and trigger control loop."""
        self._imu = msg
        # Trigger control loop if we have both IMU and joint state data
        if len(self._joint_state.name) > 0:
            self._tick(self._joint_state, self._imu)
    
    def _joint_states_callback(self, msg: JointState):
        """Store the latest joint state data."""
        self._joint_state = msg

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands.
        
        This method is called whenever new joint state and IMU data are available.
        It computes the policy's action and publishes the resulting joint
        commands.
        
        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data (orientation, angular velocity, acceleration)
        """
        # Reset if time jumped backwards (most likely due to sim time reset)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )
        
        # Calculate time delta since last tick
        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        # Run the control policy
        self.forward(joint_state, imu)

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.robot_joint_names

        # Compute final joint positions in policy order, then remap to robot order
        action_pos_policy = self.default_pos + self.action * self._action_scale
        action_pos_robot = action_pos_policy[self._policy_to_robot]
        self._joint_command.position = action_pos_robot.tolist()
        self._joint_command.velocity = np.zeros(12).tolist()
        self._joint_command.effort = np.zeros(12).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """Compute the policy observation vector from robot state.
        
        Constructs a 69-dimensional observation vector from robot sensor data:
        - Linear velocity (body frame)
        - Angular velocity (body frame)
        - Gravity direction (body frame)
        - Command velocity
        - Joint positions (relative to default)
        - Joint velocities
        - Previous action
        
        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data
            
        Returns:
            np.ndarray: 69-dimensional observation vector for the policy
        """
        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix
        # (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z
        ])
        
        # Simple integration to estimate velocity
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b
        
        # Extract angular velocity
        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])
        
        # Calculate gravity direction in body frame
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Initialize observation vector
        obs = np.zeros(48)
        
        # Fill observation vector components:
        # Base linear velocity (3)
        obs[:3] = self._lin_vel_b

        # Base angular velocity (3)
        obs[3:6] = ang_vel_b

        # Gravity direction (3)
        obs[6:9] = gravity_b
        
        # Velocity commands (3)
        cmd_vel = [
            # self._cmd_vel.linear.x,
            # 0.5,
            self._cmd_vel.linear.x,
            self._cmd_vel.linear.y,
            self._cmd_vel.angular.z
        ]
        obs[9:12] = np.array(cmd_vel)
        
        # Joint states (12 positions + 12 velocities) in policy order
        current_joint_pos = np.zeros(12)
        current_joint_vel = np.zeros(12)

        # Map joint states from ROS message to policy-ordered arrays
        for i, name in enumerate(self.policy_joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        # Store joint positions relative to default pose
        obs[12:24] = current_joint_pos - self.default_pos
        
        # Store joint velocities
        obs[24:36] = current_joint_vel
        
        # Store previous actions
        obs[36:48] = self._previous_action


        # self._logger.infor(
        #     f'{self._get_stamp_prefix()} test'
        # )

        return obs

    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation.
        
        Args:
            obs: Observation vector containing robot state information
            
        Returns:
            np.ndarray: Action vector containing joint position adjustments
        """
        # Run inference with the PyTorch policy
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions.
        
        This combines observation computation and policy evaluation.
        The policy is run at a reduced rate (decimation) to save computation.
        
        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data
        """
        # Compute observation from current state
        obs = self._compute_observation(joint_state, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Input quaternion (w, x, y, z).

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        """Load the neural network policy from the specified path."""
        # Load policy from file to io.BytesIO object
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        # Load TorchScript model from buffer
        self.policy = torch.jit.load(buffer)

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging with both system and ROS time.
        
        Returns:
            str: Formatted timestamp string with system and ROS time
        """
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds.
        
        Args:
            header: ROS message header containing timestamp
            
        Returns:
            float: Time in seconds
        """
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the H1 fullbody controller node."""
    rclpy.init(args=args)
    node = QuadroController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()