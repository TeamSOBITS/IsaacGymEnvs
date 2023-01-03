# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
# from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class SobitProBalance(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.action_speed_scale = self.cfg["env"]["actionSpeedScale"]

        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]
        self.robot_dof_noise = self.cfg["env"]["robotDofNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        # self.debug_viz = True

        # dimensions
        # obs = ["base_linvel",
        #        "base_angvel",
        #        "arm_pos", 
        #        "arm_vel",
        #        "arm_torque",
        #        "steer_pos",
        #        "wheel_vel",
        #        "ball_pos",
        #        "ball_quat",
        #        "ball_linvel"]
        # 3+3 + 5+5+5 + 4+4 + 3+4+3 = 39
        self.cfg["env"]["numObservations"] = 3+3 + 5+5+5 + 4+4 + 3+4+3
        # Actions: robot arm Dof (5)
        self.cfg["env"]["numActions"] = 5


        # Values to be filled in at runtime
        self.states = {}                    # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                   # will be dict mapping names to relevant sim handles
        self.num_dofs = None                # Total number of DOFs per env
        self.actions = None                 # Current actions to be deployed
        self._robot_id = None                # Actor ID corresponding to cubeA for a given env
        self._init_ball_state = None        # Initial state of cubeA for the current env
        self._ball_state = None             # Current state of cubeA for the current env
        self._ball_radius = None
        self._ball_id = None                # Actor ID corresponding to cubeA for a given env
        self._init_tray_state = None        # Current state of cubeA for the current env
        self._tray_state = None             # Current state of cubeA for the current env
        self._init_base_state = None        # Current state of cubeA for the current env
        self._base_state = None             # Current state of cubeA for the current env

        # Tensor placeholders
        self._root_state = None             # State of root body  (n_envs, 13)
        self._init_root_state = None
        self._rigid_body_state = None       # State of all rigid bodies (n_envs, n_bodies, 13)
        self._dof_state = None              # State of all joints (n_envs, n_dof)
        self._init_dof_state = None
        self._dof_sensor = None             # State of all joints (n_envs, n_dof)

        self._dof_positions = None          # Joint positions     (n_envs, n_dof)
        self._dof_velocities = None         # Joint velocities    (n_envs, n_dof)
        self._dof_position_targets = None
        self._dof_velocity_targets = None

        self._arm_control = None            # Tensor buffer for controlling arm
        self._pos_control = None            # Position actions
        self._vel_control = None            # Position actions
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self.num_robot_bodies = None
        self.num_robot_dofs = None
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.up_axis = "z"
        self.up_axis_idx = 2

        self.is_init = False


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Robot initial DoF pose: (arm) -> pi/2(0), -pi/2(1), 0(2), pi/2(3), 0(4); (bot)->0,0,0,0,0,0,0,0
        self.robot_default_dof_pos = to_torch(
            # 0    1    2    3    4         5      6        7      8        9      10        11     12
            [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/4.0, 0.0, np.pi/4.0, 0.0, np.pi/4.0, 0.0, -np.pi/4.0, 0.0], device=self.device
            # [math.pi/2, -math.pi/2, 0.0, math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.robot_default_dof_vel_pos = to_torch(
            # 0    1    2    3    4    5    6    7    8    9    10   11   12
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device
        )
        # Refresh tensors
        self._refresh()
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0.0
        self.sim_params.gravity.y = 0.0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/sobit_pro_tray_new.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # load robot asset
        robot_options = gymapi.AssetOptions()
        # robot_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        robot_options.flip_visual_attachments = False
        robot_options.fix_base_link = True
        robot_options.collapse_fixed_joints = False
        robot_options.disable_gravity = False
        robot_options.override_inertia = True
        # robot_options.thickness = 0.001
        robot_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_options.use_mesh_materials = True
        robot_options.slices_per_cylinder = 40
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, robot_options)

        # create ball asset
        self._ball_radius = 0.05
        ball_options = gymapi.AssetOptions()
        ball_options.density = 500
        ball_asset = self.gym.create_sphere(self.sim, self._ball_radius, ball_options)
        ball_color = gymapi.Vec3(0.6, 0.1, 0.0)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot DoF properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        robot_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        robot_dof_props['driveMode'][6::2] = gymapi.DOF_MODE_VEL

        robot_dof_stiffness = to_torch([5000., 5000., 5000., 5000., 5000., 5000., 0., 5000., 0., 5000., 0., 5000., 0.], dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 600., 1.0e2, 600., 1.0e2, 600., 1.0e2, 600.], dtype=torch.float, device=self.device)

        for i in range(self.num_robot_dofs):
            robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
            robot_dof_props['damping'][i] = robot_dof_damping[i]

            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)

        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for ball
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.robots = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )

            # Create robot
            self._robot_id = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, self._robot_id, robot_dof_props)

            # Create ball
            self._ball_id = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, "ball", i, 1, 0)
            # self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._ball_id, 0, gymapi.MESH_VISUAL, ball_color)
            # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(self._robot_id)

        # Setup init state buffer
        self._init_ball_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_tray_state = torch.zeros(self.num_envs, 13, device=self.device)
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        # self._robot_id = 0  # robot is created the first
        self.handles = {
            # Robot main links handle
            "base": self.gym.find_actor_rigid_body_handle(env_ptr, self._robot_id, "base_link"),
            "tray": self.gym.find_actor_rigid_body_handle(env_ptr, self._robot_id, "tray_tf_link"),
            # "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, self._robot_id, "panda_grip_site"),
            # Ball handle
            "ball_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._ball_id, "sphere"),
            # "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),

        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _dof_state_tensor        = self.gym.acquire_dof_state_tensor(self.sim)
        _dof_sensor_tensor       = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state       = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state        = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._dof_sensor       = gymtorch.wrap_tensor(_dof_sensor_tensor).view(self.num_envs, -1)

        self._dof_positions  = self._dof_state[..., 0]
        self._dof_velocities = self._dof_state[..., 1]

        self._base_state = self._rigid_body_state[:, self.handles["base"], :]
        self._tray_state = self._rigid_body_state[:, self.handles["tray"], :]
        self._ball_state = self._root_state[:, self._ball_id, :]
        # self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "ball_size": torch.ones_like(self._tray_state[:, 0]) * self._ball_radius,
            # "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._pos_control[:, :5]
        # self._arm_control = self._effort_control[:, :7]
        # self._gripper_control = self._pos_control[:, 7:9]

        # Initialize Initial States
        # self._init_dof_state = self._dof_state.clone()
        # self._init_root_state = self._root_state.clone()

        # Initialize indices: multiply number of agantes per env (robot+ball)
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        # Initialize targets
        self._dof_position_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self._dof_velocity_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)

    def _update_states(self):
        self.states.update({
            # Robot
            "dof_pos": self._dof_positions[:, :],
            "dof_vel": self._dof_velocities[:, :],
            "dof_torque": self._dof_sensor[:, :],
            "arm_pos": self._dof_positions[:, 0:5], # Joint positions  (n_envs, n_dof)
            "arm_vel": self._dof_velocities[:, 0:5], # Joint positions  (n_envs, n_dof)
            "arm_torque": self._dof_sensor[:, 0:5], # Joint positions  (n_envs, n_dof)
            "steer_pos": self._dof_positions[:, 5::2], # Joint positions  (n_envs, n_dof)
            "wheel_vel": self._dof_velocities[:, 6::2], # Joint positions  (n_envs, n_dof)
            # Ball
            "ball_pos": self._ball_state[:, 0:3],
            "ball_quat": self._ball_state[:, 3:7],
            "ball_linvel": self._ball_state[:, 7:10],
            "ball_angvel": self._ball_state[:, 10:13],
            # "cubeB_pos": self._cubeB_state[:, :3],
            # "cubeB_quat": self._cubeB_state[:, 3:7],
            # Tray
            "tray_pos": self._tray_state[:, 0:3],
            "tray_quat": self._tray_state[:, 3:7],
            "ball_pos_to_tray": self._ball_state[:, :3] - self._tray_state[:, :3],
            # Base
            "base_pos": self._base_state[:, 0:3],
            "base_quat": self._base_state[:, 3:7],
            "base_linvel": self._base_state[:, 7:10],
            "base_angvel": self._base_state[:, 10:13],
            "ball_pos_to_base": self._ball_state[:, :3] - self._base_state[:, :3],
            "init_tray_pos_to_base": self._init_tray_state[:, :3] - self._base_state[:, :3],
        })

        if self.is_init is False:
            self._init_tray_state = self.states["tray_pos"].clone()
            self.is_init = True
            print("self.is_init:", self.is_init)

    def _refresh(self):
        # Refresh info
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs = ["base_linvel",
               "base_angvel",
               "arm_pos", 
               "arm_vel",
               "arm_torque",
               "steer_pos",
               "wheel_vel",
               "ball_pos",
               "ball_quat",
               "ball_linvel"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_robot_reward(
            self.states,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)

        # Reset elements state: ball
        self._reset_init_elem_state(elem='ball', env_ids=env_ids, check_valid=False)
        # self._reset_init_elem_state(elem='box', env_ids=env_ids, check_valid=True)

        # Write these new init states to the sim states
        self._ball_state[env_ids] = self._init_ball_state[env_ids]
        # self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset robot
        reset_noise = torch.rand((len(env_ids), 13), device=self.device)
        pos = tensor_clamp(
            self.robot_default_dof_pos.unsqueeze(0) +
            self.robot_dof_noise * 2.0 * (reset_noise - 0.5),
            self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)
        # Overwrite move_base init pos (no noise)
        pos[:, 5:] = self.robot_default_dof_pos[5:]
            
        # Reset the internal obs accordingly
        self._dof_positions[env_ids, :] = pos
        self._dof_velocities[env_ids, :] = torch.zeros_like(self._dof_velocities[env_ids])
        # TODO set velocity!!!!!!!!!!

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update ball states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32),
            len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_elem_state(self, elem, env_ids, check_valid=True):
        """
        Simple method to sample @elem's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other elem.

        Args:
            elem(str): Which elem to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset elem for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other elem.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_elem_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if elem.lower() == 'ball':
            this_elem_state_all = self._init_ball_state
            # other_elem_state = self._init_elemB_state[env_ids, :]
            elem_heights = self.states["ball_size"]
        elif elem.lower() == 'b':
            this_elem_state_all = self._init_elemB_state
            other_elem_state = self._init_elemA_state[env_ids, :]
            elem_heights = self.states["elemA_size"]
        else:
            raise ValueError(f"Invalid elem specified, options are 'ball' and 'B'; got: {elem}")

        # Minimum elem distance for guarenteed collision-free sampling is the sum of each elem's effective radius
        # min_dists = (self.states["elemA_size"] + self.states["elemB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the elems aren't too close together
        # min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of tray TEST
        centered_elem_xy_state = self._init_tray_state[env_ids, :2]
        # centered_elem_xy_state = self._tray_state[env_ids, :2]
        # centered_elem_xy_state = torch.tensor(self._tray_state[:, 0:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_elem_state[:, 2] = self._init_tray_state[env_ids, 2] + elem_heights.squeeze(-1)[env_ids]
        # sampled_elem_state[:, 2] = self._tray_state[env_ids, 2] + elem_heights.squeeze(-1)[env_ids]

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_elem_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on elems' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_elem_state[active_idx, :2] = centered_elem_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_elem_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                elem_dist = torch.linalg.norm(sampled_elem_state[:, :2] - other_elem_state[:, :2], dim=-1)
                # active_idx = torch.nonzero(elem_dist < min_dists, as_tuple=True)[0]
                active_idx = torch.nonzero(elem_dist < 0.5, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling elem locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_elem_state[:, :2] = centered_elem_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_elem_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_elem_state[:, 3:7])

        # Lastly, set these sampled values as the new init state TEST
        this_elem_state_all[env_ids, :] = sampled_elem_state
        # self._init_ball_state[env_ids, :] = sampled_elem_state

    def pre_physics_step(self, _actions):
        self.actions = _actions.clone().to(self.device)

        actuated_idx = torch.LongTensor([0,1,2,3,4])

        # update position targets from actions
        self._dof_position_targets[..., actuated_idx] += self.dt * self.action_speed_scale * self.actions
        self._dof_position_targets[:] = tensor_clamp(self._dof_position_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits) 
        
        # Control arm (scale value first)
        # self._arm_control[:, :] += self.dt * self.action_speed_scale * self.actions
        # self._arm_control[:] = tensor_clamp(self._arm_control, self.robot_dof_lower_limits[:5], self.robot_dof_upper_limits[:5])
        # self._arm_control[:] = 0.0

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_position_targets))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            ball_pos = self.states["ball_pos"]
            ball_rot = self.states["ball_quat"]
            tray_pos = self.states["tray_pos"]
            tray_rot = self.states["tray_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((ball_pos, tray_pos), (ball_rot, tray_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_robot_reward(states, reset_buf, progress_buf, max_episode_length):
    # type: (Dict[str, torch.Tensor], Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.7)
    # ball_pos = np.zeros(states["ball_pos"].size(), dtype=gymapi.Transform.dtype)
    # ball_pos = ((states["ball_pos"]), (states["ball_quat"]))
    # ball_pos = states["tray_quat"].transform_points(states["ball_pos"])

    # ball_dist = torch.sqrt((states["ball_pos"][:, 0] - 0.944) * (states["ball_pos"][:, 0] - 0.944) +
    #                        (states["ball_pos"][:, 1]) * (states["ball_pos"][:, 1]) +
    #                        (states["ball_pos"][:, 2] - 0.89) * (states["ball_pos"][:, 2] - 0.89))
    ball_dist = torch.sqrt((states["ball_pos_to_base"][:, 0]-states["init_tray_pos_to_base"][:, 0]) * (states["ball_pos_to_base"][:, 0]-states["init_tray_pos_to_base"][:, 0]) +
                           (states["ball_pos_to_base"][:, 1]-states["init_tray_pos_to_base"][:, 1]) * (states["ball_pos_to_base"][:, 1]-states["init_tray_pos_to_base"][:, 1]) +
                           (states["ball_pos_to_base"][:, 2]-states["init_tray_pos_to_base"][:, 2]-states["ball_size"][:]) * (states["ball_pos_to_base"][:, 2]-states["init_tray_pos_to_base"][:, 2]-states["ball_size"][:]))
    ball_speed = torch.sqrt(states["ball_linvel"][:, 0] * states["ball_linvel"][:, 0] +
                            states["ball_linvel"][:, 1] * states["ball_linvel"][:, 1] +
                            states["ball_linvel"][:, 2] * states["ball_linvel"][:, 2])
    tray_orien = torch.sqrt(states["tray_quat"][:, 0] * states["base_quat"][:, 0] +
                            states["tray_quat"][:, 1] * states["base_quat"][:, 1] +
                            states["tray_quat"][:, 2] * states["base_quat"][:, 2] +
                            (states["tray_quat"][:, 3]-1.0) * (states["base_quat"][:, 3]-1.0))
    # print("x ",states["ball_pos_to_tray"][0, 0] * states["ball_pos_to_tray"][0, 0])
    # print("y ",states["ball_pos_to_tray"][0, 1] * states["ball_pos_to_tray"][0, 1])
    # print("z ",(states["ball_pos_to_tray"][0, 2]-states["ball_size"][0]) * (states["ball_pos_to_tray"][0, 2]-states["ball_size"][0]))
    # tray_rotation = torch.sqrt(dof_positions[...,3] * dof_positions[...,3])
    pos_reward = 1.0 / (1.0 + ball_dist)
    speed_reward = 1.0 / (1.0 + ball_speed)
    reward = pos_reward * speed_reward + tray_orien

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where((states["ball_pos"][:, 2] < states["tray_pos"][:, 2]) | (states["ball_pos"][:, 2] > 0.01+states["ball_size"]+states["tray_pos"][:, 2]), torch.ones_like(reset_buf), reset)
    reset = torch.where(states["ball_pos"][:, 2] < states["tray_pos"][:, 2], torch.ones_like(reset_buf), reset)

    return reward, reset
