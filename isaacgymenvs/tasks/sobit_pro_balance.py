# Copyright (c) 2021-2022, NVIDIA Corporation
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

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
        print("here")
        self.cfg = cfg

        # Params from config file
        self.max_episode_length   = self.cfg["env"]["maxEpisodeLength"]

        self.action_scale         = self.cfg["env"]["actionSpeedScale"]

        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]

        self.robot_dof_noise      = self.cfg["env"]["robotDofNoise"]

        # dimensions
        # obs include: 
        # || base_linvel (3) + base_angvel (3) + 
        # || dof_pos_arm (5) + dof_vel_arm (5) + dof_force_arm (5) + dof_pos_steer (4) + dof_vel_wheel (4) +
        # || tray_pos (3) + tray_quat (4) +
        # || ball_to_base_pos (3) + ball_quat (4) + ball_linvel (3)
        # || cubeA_pos (3) + cubeA_quat (4) + cubeA_linvel (3) +
        # || cubeB_pos (3) + cubeB_quat (4) + cubeB_linvel (3) 
        # || TOTAL: 66
        self.cfg["env"]["numObservations"] = (3+3) + (5+5+5+4+4) + (3+4) + (3+4+3) + (3+4+3) + (3+4+3)
        # actions include: joint torques (5)
        self.cfg["env"]["numActions"] = 5

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        self._init_robot_state = None           # Initial state of robot for the current env
        self._init_ball_state = None           # Initial state of cubeB for the current env
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env

        self._robot_state = None                # Current state of robot for the current env
        self._ball_state = None                # Current state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env

        self._robot_id = None                   # Actor ID corresponding to cubeA for a given env
        self._ball_id = None                    # Actor ID corresponding to cubeB for a given env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)

        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._dof_sensor = None                 # Forces of all joints      (n_envs, n_dof)

        self._dof_positions = None              # Joint positions           (n_envs, n_dof)
        self._dof_velocities = None             # Joint velocities          (n_envs, n_dof)
        self._dof_forces = None                 # Forces velocities         (n_envs, n_dof)

        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._base_state = None
        self._tray_state = None

        self._arm_control = None                # Tensor buffer for controlling arm
        self._steer_control = None              # Tensor buffer for controlling gripper
        self._wheel_control = None              # Tensor buffer for controlling gripper
        self._pos_control = None                # Position actions
        self._vel_control = None                # Velocity actions
        self._effort_control = None             # Torque actions

        self.robot_dof_effort_limits = None     # Actuator effort limits for robot
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        # Debug
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Environment
        # self.up_axis = self.cfg["sim"]["up_axis"]
        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Robot default positions
        self.robot_default_dof_pos = to_torch(
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device
            [0, 0, 0, 0, 0, -np.pi/4, 0, np.pi/4, 0, np.pi/4, 0, -np.pi/4, 0], device=self.device
        )
        vel = 10.0
        self.robot_default_dof_vel = to_torch(
            # 0    1    2    3    4    5     6    7    8    9     10   11   12
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -vel, 0.0, vel, 0.0, -vel, 0.0, vel], device=self.device
        )

        # Set control limits
        self.cmd_limit = self.robot_dof_effort_limits[:5].unsqueeze(0)

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
        env_lower = gymapi.Vec3(-spacing, -spacing/spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing/spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/sobit_pro_tray_new.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # load robot asset
        robot_opts = gymapi.AssetOptions()
        robot_opts.flip_visual_attachments = False
        robot_opts.fix_base_link = False
        robot_opts.collapse_fixed_joints = False
        robot_opts.disable_gravity = False
        robot_opts.thickness = 0.001
        robot_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_opts.use_mesh_materials = True
        # robot_opts.slices_per_cylinder = 40
        robot_opts.replace_cylinder_with_capsule = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, robot_opts)

        robot_dof_stiffness = to_torch([1000., 1000., 1000., 1000., 1000., 1000., 0., 1000., 0., 1000., 0., 1000., 0.], dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([100, 100, 100, 100, 100, 100, 600., 100, 600., 100, 600., 100, 600.], dtype=torch.float, device=self.device)

        # Objects size
        self.cubeA_size = 0.050
        self.cubeB_size = 0.070
        self.ball_size = 0.030


        # Create ball asset
        ball_opts  = gymapi.AssetOptions()
        ball_opts.density  = 1.0
        ball_asset = self.gym.create_sphere(self.sim, self.ball_size, ball_opts)
        ball_color = gymapi.Vec3(1.0, 0.0, 0.0)

        # Create cubeA asset
        cubeA_opts  = gymapi.AssetOptions()
        cubeA_opts.density  = 1.0
        cubeA_asset = self.gym.create_box(self.sim, self.cubeA_size,self.cubeA_size,2*self.cubeA_size, cubeA_opts)
        cubeA_color = gymapi.Vec3(1.0, 1.0, 0.0)

        # Create cubeB asset
        cubeB_opts  = gymapi.AssetOptions()
        cubeB_opts.density  = 1.0
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.0, 1.0)


        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs   = self.gym.get_asset_dof_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits  = []
        self.robot_dof_upper_limits  = []
        self.robot_dof_effort_limits = []
        robot_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        robot_dof_props['driveMode'][6::2] = gymapi.DOF_MODE_VEL


        for i in range(self.num_robot_dofs):

            robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
            robot_dof_props['damping'][i]   = robot_dof_damping[i]

            robot_dof_props['effort'][i] = 100

            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self.robot_dof_effort_limits.append(robot_dof_props['effort'][i])

        self.robot_dof_lower_limits  = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits  = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_effort_limits = to_torch(self.robot_dof_effort_limits, device=self.device)

        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Tray start pose
        tray_pos = [0.95, 0.0, 0.90]
        self._tray_surface_pos = np.array(tray_pos)

        # Define start pose for elements (doesn't really matter since they're get overridden during reset() anyways)
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        self.robots = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # Create robot
            # NOTE: robot should ALWAYS be loaded first in sim!
            # Randomize start pose (if noise is set)
            if self.robot_position_noise > 0:
                rand_xy = self.robot_position_noise * (-1. + np.random.rand(2) * 2.0)
                robot_start_pose.p = gymapi.Vec3(0.0 + rand_xy[0], 0.0 + rand_xy[1], 0.0)
            if self.robot_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.robot_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_dof_positionsuat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                robot_start_pose.r = gymapi.Quat(*new_dof_positionsuat)
            
            self._robot_id = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, self._robot_id, robot_dof_props)

            # Create cubes
            self._ball_id  = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, "ball", i, -1, 0)
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, -1, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, -1, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._ball_id, 0, gymapi.MESH_VISUAL, ball_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(self._robot_id)

        # Setup init state buffer
        self._init_robot_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_robot_state[:,6] = 1
        self._init_ball_state  = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0
        self.handles = {
            # Robot
            "base": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "base_footprint"),
            "tray": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "tray_tf_link"),
            # Cubes
            "robot_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._robot_id, "ball"),
            "ball_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._ball_id, "ball"),
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _dof_sensor_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._dof_sensor = gymtorch.wrap_tensor(_dof_sensor_tensor).view(self.num_envs, -1)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        
        self._dof_positions = self._dof_state[..., 0]
        self._dof_velocities = self._dof_state[..., 1]

        self._base_state = self._rigid_body_state[:, self.handles["base"], :]
        self._tray_state = self._rigid_body_state[:, self.handles["tray"], :]

        self._robot_state = self._root_state[:, self._robot_id, :]
        self._ball_state  = self._root_state[:, self._ball_id, :]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "ball_size" : torch.ones_like(self._tray_state[:, 0]) * self.ball_size,
            "cubeA_size": torch.ones_like(self._tray_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._tray_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        self._pos_control    = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._vel_control    = torch.zeros_like(self._pos_control)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control   = self._effort_control[:, :5]
        self._arm_control   = self._pos_control[:, :5]
        self._steer_control = self._pos_control[:, 5::2]
        self._wheel_control = self._vel_control[:, 6::2]

        # Initialize buffers
        self.extras = {}

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Robot
            "dof_pos"      : self._dof_positions[:, :],
            "dof_vel"      : self._dof_velocities[:, :],
            "dof_force"    : self._dof_sensor[:, :],
            "dof_pos_arm"  : self._dof_positions[:, :5],
            "dof_vel_arm"  : self._dof_velocities[:, :5],
            "dof_pos_steer": self._dof_positions[:, 5::2],
            "dof_vel_wheel": self._dof_velocities[:, 6::2],
            "dof_force_arm": self._dof_sensor[:, :5],
            "robot_pos"   : self._robot_state[:, :3],
            "robot_quat"  : self._robot_state[:, 3:7],
            "robot_linvel": self._robot_state[:, 7:10],
            "robot_angvel": self._robot_state[:, 10:13],
            # Base
            "base_pos"   : self._base_state[:, :3],
            "base_quat"  : self._base_state[:, 3:7],
            "base_linvel": self._base_state[:, 7:10],
            "base_angvel": self._base_state[:, 10:13],
            # Tray
            "tray_pos"   : self._tray_state[:, :3],
            "tray_quat"  : self._tray_state[:, 3:7],
            "tray_linvel": self._tray_state[:, 7:10],
            "tray_angvel": self._tray_state[:, 10:13],
            "tray_to_base_pos": self._tray_state[:, :3] - self._base_state[:, :3],
            # Ball
            "ball_pos":    self._ball_state[:, :3],
            "ball_quat":   self._ball_state[:, 3:7],
            "ball_linvel": self._ball_state[:, 7:10],
            "ball_angvel": self._ball_state[:, 10:13],
            "ball_to_tray_pos": self._ball_state[:, :3] - self._tray_state[:, :3],
            "ball_to_base_pos": self._ball_state[:, :3] - self._base_state[:, :3],
            # Cubes
            "cubeA_pos":    self._cubeA_state[:, :3],
            "cubeA_quat":   self._cubeA_state[:, 3:7],
            "cubeA_linvel": self._cubeA_state[:, 7:10],
            "cubeA_angvel": self._cubeA_state[:, 10:13],
            "cubeA_to_tray_pos": self._cubeA_state[:, :3] - self._tray_state[:, :3],
            "cubeA_to_base_pos": self._cubeA_state[:, :3] - self._base_state[:, :3],
            "cubeB_pos":    self._cubeB_state[:, :3],
            "cubeB_quat":   self._cubeB_state[:, 3:7],
            "cubeB_linvel": self._cubeB_state[:, 7:10],
            "cubeB_angvel": self._cubeB_state[:, 10:13],
            "cubeB_to_tray_pos": self._cubeB_state[:, :3] - self._tray_state[:, :3],
            "cubeB_to_base_pos": self._cubeB_state[:, :3] - self._base_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.vel[:] = compute_robot_reward(
            self.states,
            self.reset_buf, self.progress_buf, self.actions, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["base_linvel",
               "base_angvel",
               "dof_pos_arm", 
               "dof_vel_arm",
               "dof_force_arm",
               "dof_pos_steer",
               "dof_vel_wheel",
               "tray_to_base_pos",
               "tray_quat",
               "ball_to_base_pos",
               "ball_quat",
               "ball_linvel",
               "cubeA_to_base_pos",
               "cubeA_quat",
               "cubeA_linvel",
               "cubeB_to_base_pos",
               "cubeB_quat",
               "cubeB_linvel"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset elements and robot positions
        self._reset_init_elem_state(elem='ball' , env_ids=env_ids, check_valid=False)
        self._reset_init_elem_state(elem='cubeA', env_ids=env_ids, check_valid=True)
        self._reset_init_elem_state(elem='cubeB', env_ids=env_ids, check_valid=False)

        # Write these new init states to the sim states
        self._robot_state[env_ids] = self._init_robot_state[env_ids]
        self._ball_state[env_ids]  = self._init_ball_state[env_ids]
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 13), device=self.device)
        robot_reset_dof_pos = tensor_clamp(
            self.robot_default_dof_pos.unsqueeze(0) +
            self.robot_dof_noise * 2.0 * (reset_noise - 0.5),
            self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)

        # Overwrite steer and wheel init pos (no noise since these are always position controlled)
        robot_reset_dof_pos[:, 5:] = self.robot_default_dof_pos[5:]

        # Reset the internal obs accordingly
        self._dof_positions[env_ids, :] = robot_reset_dof_pos
        self._dof_velocities[env_ids, :] = torch.zeros_like(self._dof_velocities[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = robot_reset_dof_pos
        self._vel_control[env_ids, :] = torch.zeros_like(robot_reset_dof_pos)
        # self._effort_control[env_ids, :] = torch.zeros_like(robot_reset_dof_pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._vel_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_elem_int32 = self._global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_elem_int32), len(multi_env_ids_elem_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.vel[env_ids] = 0

    def _reset_init_elem_state(self, elem, env_ids, check_valid=True):
        """
        Simple method to sample @elem's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_elemX_state

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
        # sampled_robot_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if elem.lower() == 'ball':
            this_elem_state_all = self._init_ball_state
            other_elem1_state = self._init_cubeA_state[env_ids, :]
            other_elem2_state = self._init_cubeB_state[env_ids, :]
            elem_heights = self.states["ball_size"].squeeze(-1)[env_ids]
        elif elem.lower() == 'cubea':
            this_elem_state_all = self._init_cubeA_state
            other_elem1_state = self._init_ball_state[env_ids, :]
            other_elem2_state = self._init_cubeB_state[env_ids, :]
            elem_heights = self.states["cubeA_size"].squeeze(-1)[env_ids] / 2
        elif elem.lower() == 'cubeb':
            this_elem_state_all = self._init_cubeB_state
            other_elem1_state = self._init_ball_state[env_ids, :]
            other_elem2_state = self._init_cubeA_state[env_ids, :]
            elem_heights = self.states["cubeB_size"].squeeze(-1)[env_ids] / 2
        else:
            raise ValueError(f"Invalid cube specified, options are 'ball', 'cubeA' and 'cubeB'; got: {elem}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_1_dists = (self.states["cubeA_size"] + self.states["ball_size"])[env_ids] / 3.0
        min_2_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] / 3.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_1_dists = min_1_dists * 2.0
        min_2_dists = min_2_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_elem_xy_state = torch.tensor(self._tray_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_elem_state[:, 2] = self._tray_surface_pos[2] + elem_heights

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_elem_state[:, 6] = 1.0
        # sampled_robot_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
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
                cube_to_ball_dist = torch.linalg.norm(sampled_elem_state[:, :2] - other_elem1_state[:, :2], dim=-1)
                cube_to_cube_dist = torch.linalg.norm(sampled_elem_state[:, :2] - other_elem2_state[:, :2], dim=-1)
                active_1_idx = torch.nonzero(cube_to_ball_dist < min_1_dists, as_tuple=True)[0]
                active_2_idx = torch.nonzero((cube_to_cube_dist < min_2_dists) & (cube_to_ball_dist < min_1_dists), as_tuple=True)[0]
                if elem.lower() == 'cubea':
                    num_active_idx = len(active_1_idx)
                elif elem.lower() == 'cubeb':
                    num_active_idx = len(active_1_idx) + len(active_2_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
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

        # Lastly, set these sampled values as the new init state
        this_elem_state_all[env_ids, :] = sampled_elem_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        # u_arm = self.actions[:, :5]

        u_arm = self.dt * self.action_scale * self.actions

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        # self._arm_control[:, :] = u_arm
        self._arm_control[:, :] = u_arm
        self._arm_control[:, :] = tensor_clamp(self._arm_control.unsqueeze(0), 
                                            self.robot_dof_lower_limits[:5].unsqueeze(0), self.robot_dof_upper_limits[:5].unsqueeze(0))
        self._arm_control[:, :] = torch.zeros_like(self._arm_control[:, :])

        # Control gripper
        # Write gripper command to appropriate tensor buffer
        self._steer_control[:, :] = self.robot_default_dof_pos[5::2]
        self._wheel_control[:, :] = self.robot_default_dof_vel[6::2]

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            tray_pos  = self.states["tray_pos"]
            tray_rot  = self.states["tray_quat"]
            ball_pos = self.states["ball_pos"]
            ball_rot = self.states["ball_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((tray_pos, ball_pos, cubeA_pos, cubeB_pos), (tray_rot, ball_rot, cubeA_rot, cubeB_rot)):
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
def compute_robot_reward(
    states, reset_buf, progress_buf, actions, max_episode_length
):
    # type: (Dict[str, Tensor], Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.944)
    # ball_pos = torch.sqrt((states["ball_to_base_pos"][:, 0]-0.944) * (states["ball_to_base_pos"][:, 0]-0.944) +
    #                       (states["ball_to_base_pos"][:, 1]) * (states["ball_to_base_pos"][:, 1]) +
    #                       (states["ball_to_base_pos"][:, 2]-states["ball_size"][:]-0.89) * (states["ball_to_base_pos"][:, 2]-states["ball_size"][:]-0.89))
    # ball_quat = torch.sqrt(states["ball_quat"][:, 0] * states["ball_quat"][:, 0] +
    #                        states["ball_quat"][:, 1] * states["ball_quat"][:, 1] +
    #                        states["ball_quat"][:, 2] * states["ball_quat"][:, 2])
    ball_pos = torch.sqrt((states["ball_to_base_pos"][:, 0]-0.944) * (states["ball_to_base_pos"][:, 0]-0.944) +
                          (states["ball_to_base_pos"][:, 1]) * (states["ball_to_base_pos"][:, 1]) +
                          (states["ball_to_base_pos"][:, 2]-states["ball_size"][:]-0.89) * (states["ball_to_base_pos"][:, 2]-states["ball_size"][:]-0.89) + 
                           states["ball_quat"][:, 0] * states["ball_quat"][:, 0] +
                           states["ball_quat"][:, 1] * states["ball_quat"][:, 1] +
                           states["ball_quat"][:, 2] * states["ball_quat"][:, 2])
    ball_speed = torch.sqrt((states["ball_linvel"][:, 0]-states["base_linvel"][:, 0]) * (states["ball_linvel"][:, 0]-states["base_linvel"][:, 0]) +
                            (states["ball_linvel"][:, 1]-states["base_linvel"][:, 1]) * (states["ball_linvel"][:, 1]-states["base_linvel"][:, 1]) +
                            (states["ball_linvel"][:, 2]-states["base_linvel"][:, 2]) * (states["ball_linvel"][:, 2]-states["base_linvel"][:, 2]))

    # cubeA_pos = torch.sqrt((states["cubeA_to_base_pos"][:, 0]-0.944) * (states["cubeA_to_base_pos"][:, 0]-0.944) +
    #                        (states["cubeA_to_base_pos"][:, 1]) * (states["cubeA_to_base_pos"][:, 1]) +
    #                        (states["cubeA_to_base_pos"][:, 2]-states["cubeA_size"][:]/2.-0.89) * (states["cubeA_to_base_pos"][:, 2]-states["cubeA_size"][:]/2.-0.89))
    # cubeA_quat = torch.sqrt(states["cubeA_quat"][:, 0] * states["cubeA_quat"][:, 0] +
    #                         states["cubeA_quat"][:, 1] * states["cubeA_quat"][:, 1] +
    #                         states["cubeA_quat"][:, 2] * states["cubeA_quat"][:, 2])
    cubeA_pos = torch.sqrt((states["cubeA_to_base_pos"][:, 0]-0.944) * (states["cubeA_to_base_pos"][:, 0]-0.944) +
                           (states["cubeA_to_base_pos"][:, 1]) * (states["cubeA_to_base_pos"][:, 1]) +
                           (states["cubeA_to_base_pos"][:, 2]-states["cubeA_size"][:]/2.-0.89) * (states["cubeA_to_base_pos"][:, 2]-states["cubeA_size"][:]/2.-0.89) + 
                            states["cubeA_quat"][:, 0] * states["cubeA_quat"][:, 0] +
                            states["cubeA_quat"][:, 1] * states["cubeA_quat"][:, 1] +
                            states["cubeA_quat"][:, 2] * states["cubeA_quat"][:, 2])
    cubeA_speed = torch.sqrt((states["cubeA_linvel"][:, 0]-states["base_linvel"][:, 0]) * (states["cubeA_linvel"][:, 0]-states["base_linvel"][:, 0]) +
                             (states["cubeA_linvel"][:, 1]-states["base_linvel"][:, 1]) * (states["cubeA_linvel"][:, 1]-states["base_linvel"][:, 1]) +
                             (states["cubeA_linvel"][:, 2]-states["base_linvel"][:, 2]) * (states["cubeA_linvel"][:, 2]-states["base_linvel"][:, 2]))

    # cubeB_pos = torch.sqrt((states["cubeB_to_base_pos"][:, 0]-0.944) * (states["cubeB_to_base_pos"][:, 0]-0.944) +
    #                        (states["cubeB_to_base_pos"][:, 1]) * (states["cubeB_to_base_pos"][:, 1]) +
    #                        (states["cubeB_to_base_pos"][:, 2]-states["cubeB_size"][:]/2.-0.89) * (states["cubeB_to_base_pos"][:, 2]-states["cubeB_size"][:]/2.-0.89))
    # cubeB_quat = torch.sqrt(states["cubeB_quat"][:, 0] * states["cubeB_quat"][:, 0] +
    #                         states["cubeB_quat"][:, 1] * states["cubeB_quat"][:, 1] +
    #                         states["cubeB_quat"][:, 2] * states["cubeB_quat"][:, 2])
    cubeB_pos = torch.sqrt((states["cubeB_to_base_pos"][:, 0]-0.944) * (states["cubeB_to_base_pos"][:, 0]-0.944) +
                           (states["cubeB_to_base_pos"][:, 1]) * (states["cubeB_to_base_pos"][:, 1]) +
                           (states["cubeB_to_base_pos"][:, 2]-states["cubeB_size"][:]/2.-0.89) * (states["cubeB_to_base_pos"][:, 2]-states["cubeB_size"][:]/2.-0.89) + 
                            states["cubeB_quat"][:, 0] * states["cubeB_quat"][:, 0] +
                            states["cubeB_quat"][:, 1] * states["cubeB_quat"][:, 1] +
                            states["cubeB_quat"][:, 2] * states["cubeB_quat"][:, 2])
    cubeB_speed = torch.sqrt((states["cubeB_linvel"][:, 0]-states["base_linvel"][:, 0]) * (states["cubeB_linvel"][:, 0]-states["base_linvel"][:, 0]) +
                             (states["cubeB_linvel"][:, 1]-states["base_linvel"][:, 1]) * (states["cubeB_linvel"][:, 1]-states["base_linvel"][:, 1]) +
                             (states["cubeB_linvel"][:, 2]-states["base_linvel"][:, 2]) * (states["cubeB_linvel"][:, 2]-states["base_linvel"][:, 2]))

    ball_pos_reward = 1.0 / (1.0 + ball_pos)
    # ball_quat_reward = 1.0 / (1.0 + ball_quat)
    ball_speed_reward = 1.0 / (1.0 + ball_speed)
    # ball_reward = ball_pos_reward * ball_quat_reward * ball_speed_reward
    ball_reward = ball_pos_reward * ball_speed_reward

    cubeA_pos_reward = 1.0 / (1.0 + cubeA_pos)
    # cubeA_quat_reward = 1.0 / (1.0 + cubeA_quat)
    cubeA_speed_reward = 1.0 / (1.0 + cubeA_speed)
    # cubeA_reward = cubeA_pos_reward * cubeA_quat_reward * cubeA_speed_reward
    cubeA_reward = cubeA_pos_reward * cubeA_speed_reward

    cubeB_pos_reward = 1.0 / (1.0 + cubeB_pos)
    # cubeB_quat_reward = 1.0 / (1.0 + cubeB_quat)
    cubeB_speed_reward = 1.0 / (1.0 + cubeB_speed)
    # cubeB_reward = cubeB_pos_reward * cubeB_quat_reward * cubeB_speed_reward
    cubeB_reward = cubeB_pos_reward * cubeB_speed_reward


    reward = ball_reward + cubeA_reward + cubeB_reward #* tray_reward # - pos_penalty

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where((states["ball_to_base_pos"][:, 2] < states["tray_to_base_pos"][:, 2])|(states["cubeA_to_base_pos"][:, 2] < states["tray_to_base_pos"][:, 2])|(states["cubeB_to_base_pos"][:, 2] < states["tray_to_base_pos"][:, 2]), torch.ones_like(reset_buf), reset)


    return reward, reset, ball_speed