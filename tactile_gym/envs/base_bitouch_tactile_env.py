import sys
import gym
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pkgutil
import cv2

from tactile_sim.utils.setup_pb_utils import load_standard_environment, load_standard_bitouch_environment
from tactile_sim.utils.pybullet_draw_utils import draw_frame, draw_box
from tactile_sim.utils.transforms import inv_transform_eul, transform_eul, inv_transform_vec_eul, transform_vec_eul
from ipdb import set_trace
tcp_action_mapping = {
    'x': 0, 'y': 1, 'z': 2,
    'Rx': 3, 'Ry': 4, 'Rz': 5,
}
joint_action_mapping = {
    'J1': 0, 'J2': 1, 'J3': 2,
    'J4': 3, 'J5': 4, 'J6': 5,
    'J7': 6,
}

_FASTER_MODE = True
class BaseBitouchTactileEnv(gym.Env):
    def __init__(self, env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params, if_bitouch=True):

        self.seed()

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        self._env_params = env_params
        
        self._robot_arm_params = robot_arm_params
        self._tactile_sensor_params = tactile_sensor_params
        self._visual_sensor_params = visual_sensor_params

        # env params
        self._max_steps = env_params["max_steps"]
        self._show_gui = env_params["show_gui"]
        self._observation = []
        self._env_step_counter = 0
        self._first_render = True
        self._render_closed = False

        self._workframe = self._env_params["workframe"]
        self._tcp_lims = self._env_params["tcp_lims"]

        self.connect_pybullet()
        self.set_pybullet_params()

        # set vars for full pybullet reset to clear cache
        self.reset_counter = 0
        self.reset_limit = 1000000

        self.if_bitouch = if_bitouch

    def connect_pybullet(self):
        """Connect to pybullet with/without gui enabled."""
        if self._show_gui:
            self._pb = bc.BulletClient(connection_mode=pb.GUI)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self._pb = bc.BulletClient(connection_mode=pb.DIRECT)
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                self._pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._pb.loadPlugin("eglRendererPlugin")

        # bc automatically sets client but keep here incase needed
        self._physics_client_id = self._pb._client

    def set_pybullet_params(self):
        self._pb.setGravity(0, 0, -9.81)
        self._pb.setPhysicsEngineParameter(
            fixedTimeStep=self._sim_time_step,
            numSolverIterations=150,
            enableConeFriction=1,
            contactBreakingThreshold=0.0001
        )

    def full_reset(self):
        self._pb.resetSimulation()

        load_standard_bitouch_environment(self._pb)

        self.embodiment_0.full_reset()
        self.embodiment_1.full_reset()
        self.reset_counter = 0

    def seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.close()

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
        if not self._render_closed:
            cv2.destroyAllWindows()

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.25, 0.25

        # define action ranges per act dim to rescale output of policy
        if self._robot_arm_params["control_mode"] == "tcp_position_control":
            self.lin_pos_lim = 0.001  # m
            self.ang_pos_lim = 1 * (np.pi / 180)  # rad

        elif self._robot_arm_params["control_mode"] == "tcp_velocity_control":
            if self.env_name == "bipush":
                self.lin_vel_lim = 0.01  # m/s
                self.ang_vel_lim = 5.0 * (np.pi / 180)  # rad/s
            else:
                self.lin_vel_lim = 0.005  # m/s
                self.ang_vel_lim = 10.0 * (np.pi / 180)  # rad/s

        elif self._robot_arm_params["control_mode"] == "joint_position_control":
            self.joint_pos_lim = 0.05 * (np.pi / 180)  # rad

        elif self._robot_arm_params["control_mode"] == "joint_velocity_control":
            self.joint_vel_lim = 1.0 * (np.pi / 180)  # rad/s

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_observation_space(self):

        obs_dim_dict = self.get_obs_dim()
        spaces = {
            "oracle": gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim_dict["oracle"], dtype=np.float32),
            "tactile": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["tactile"], dtype=np.uint8),
            "visual": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["visual"], dtype=np.uint8),
            "extended_feature": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict["extended_feature"], dtype=np.float32
            ),
        }

        observation_mode = self._env_params["observation_mode"]

        if observation_mode == "oracle":
            self.observation_space = gym.spaces.Dict({"oracle": spaces["oracle"]})

        elif observation_mode == "tactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"]})

        elif observation_mode == "visual":
            self.observation_space = gym.spaces.Dict({"visual": spaces["visual"]})

        elif observation_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"], "visual": spaces["visual"]})

        elif observation_mode == "tactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "extended_feature": spaces["extended_feature"]}
            )

        elif observation_mode == "visual_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

        elif observation_mode == "visuotactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

    def get_obs_dim(self):
        obs_dim_dict = {
            "oracle": self.get_oracle_obs().shape,
            "tactile": self.get_tactile_obs().shape,
            "visual": self.get_visual_obs().shape,
            "extended_feature": self.get_extended_feature_array().shape,
        }
        return obs_dim_dict

    def get_act_dim(self):
        return len(self._robot_arm_params["control_dofs"])

    def scale_actions(self, actions):
        """Scale actions from input range to range specific to actions space."""

        actions = np.clip(actions, self.min_action, self.max_action)
        input_range = self.max_action - self.min_action

        # define action ranges per act dim to rescale output of policy
        if self._robot_arm_params["control_mode"] == "tcp_position_control":
            scaled_actions = np.zeros(6)
            scaled_actions[:3] = (((actions[:3] - self.min_action) * (2*self.lin_pos_lim)) / input_range) - self.lin_pos_lim
            scaled_actions[3:] = (((actions[3:] - self.min_action) * (2*self.ang_pos_lim)) / input_range) - self.ang_pos_lim

        elif self._robot_arm_params["control_mode"] == "tcp_velocity_control":
            scaled_actions = np.zeros(6)
            scaled_actions[:3] = (((actions[:3] - self.min_action) * (2*self.lin_vel_lim)) / input_range) - self.lin_vel_lim
            scaled_actions[3:] = (((actions[3:] - self.min_action) * (2*self.ang_vel_lim)) / input_range) - self.ang_vel_lim
        elif self._robot_arm_params["control_mode"] == "joint_position_control":
            scaled_actions = (((actions - self.min_action) * (2*self.joint_pos_lim)) / input_range) - self.joint_pos_lim

        elif self._robot_arm_params["control_mode"] == "joint_velocity_control":
            scaled_actions = (((actions - self.min_action) * (2*self.joint_vel_lim)) / input_range) - self.joint_vel_lim

        return np.array(scaled_actions)

    def transform_actions(self, actions):
        """
        Converts an action defined in the workframe to an action defined in the worldframe
        """
        pass

    def step(self, action):
        """
        Encode actions, send to embodiment to be applied to the environment.
        Return observation, reward, terminal, info
        """
        if _FASTER_MODE:
            self.step_bi_touch_fast(action)
        else:
            self.step_bi_touch(action)
            
        reward, done = self.get_step_data()
        self._observation = self.get_observation()
        return self._observation, reward, done, {}
    
    def step_bi_touch(self, actions, count_step=True):
        # act
        # self.task_callback()
        if self.act_dim == 6:
            actions_list = [actions[:3], actions[3:6]]
        elif self.act_dim == 4:
            actions_list = [actions[:2], actions[2:4]]
        elif self.act_dim ==2 :
            actions_list = [actions[0], actions[1]]
        else:
            assert ("Movement mode not implement yet",self.movement_mode)
        embodiments_list = [self.embodiment_0, self.embodiment_1]
        # scale and embed actions appropriately
        for embodiment, act in zip(embodiments_list, actions_list):
            encoded_action = self.encode_actions(act, embodiment)
            scaled_action = self.scale_actions(encoded_action)
            if self._robot_arm_params["control_mode"] == "tcp_velocity_control":
                clipped_action = self.check_TCP_vel_lims(embodiment, scaled_action)
                target_vel = self.workvel_to_worldvel(clipped_action)
            # print("actions:", act)
            # print("encoded_action:", encoded_action)
            # print("scaled_action:", scaled_action)
            # print("clipped_action:", clipped_action)
            self.apply_action(
                motor_commands = target_vel,
                embodiment = embodiment,
                control_mode=self._robot_arm_params["control_mode"],
                velocity_action_repeat=self._velocity_action_repeat,
                max_steps=self._max_blocking_pos_move_steps,
            )
        if count_step:
            self._env_step_counter += 1

        reward, done = self.get_step_data()

        self._observation = self.get_observation()

        # can be helpful when debugging
        # self.render()
        # time.sleep(0.01)
        return self._observation, reward, done, {}

    def apply_action(
        self,
        embodiment,
        motor_commands,
        control_mode="tcp_velocity_control",
        velocity_action_repeat=1,
        max_steps=100,
    ):
        # set the simulation with desired control points
        if control_mode == "tcp_position_control":
            embodiment.arm.set_target_tcp_pose(motor_commands)
        elif control_mode == "tcp_velocity_control":
            embodiment.arm.set_target_tcp_velocities(motor_commands)
        elif control_mode == "joint_position_control":
            embodiment.arm.set_target_joint_positions(motor_commands)
        elif control_mode == "joint_velocity_control":
            embodiment.arm.set_target_joint_velocities(motor_commands)
        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))

        # run the simulation for a number of steps
        if "position" in control_mode:
            embodiment.arm.blocking_position_move(
                max_steps=max_steps,
                constant_vel=None,
                j_pos_tol=1e-6,
                j_vel_tol=1e-3,
            )
        elif "velocity" in control_mode:
            embodiment.arm.blocking_velocity_move(blocking_steps=velocity_action_repeat)
        else:
            self.step_sim()

    def step_bi_touch_fast(self, actions, count_step=True):
        # act
        # self.task_callback()
        if self.act_dim == 6:
            actions_list = [actions[:3], actions[3:6]]
        elif self.act_dim == 4:
            actions_list = [actions[:2], actions[2:4]]
        elif self.act_dim ==2 :
            actions_list = [actions[0], actions[1]]
        else:
            assert ("Movement mode not implement yet",self.movement_mode)
        embodiments_list = [self.embodiment_0, self.embodiment_1]
        target_vels_list = []
        # scale and embed actions appropriately
        for embodiment, act in zip(embodiments_list, actions_list):
            encoded_action = self.encode_actions(act, embodiment)
            scaled_action = self.scale_actions(encoded_action)
            if self._robot_arm_params["control_mode"] == "tcp_velocity_control":
                clipped_action = self.check_TCP_vel_lims(embodiment, scaled_action)
                target_vel = self.workvel_to_worldvel(clipped_action)
                target_vels_list.append(target_vel)
            else:
                assert ("Control mode not implement yet for Bi-Touch:",self._robot_arm_params["control_mode"])
            # print("actions:", act)
            # print("encoded_action:", encoded_action)
            # print("scaled_action:", scaled_action)
            # print("clipped_action:", clipped_action)
        self.apply_action_bitouch(
            motor_commands = target_vels_list,
            control_mode=self._robot_arm_params["control_mode"],
            velocity_action_repeat=self._velocity_action_repeat,
        )
        
        if count_step:
            self._env_step_counter += 1

        reward, done = self.get_step_data()

        self._observation = self.get_observation()

        # can be helpful when debugging
        # self.render()
        # time.sleep(0.01)
        return self._observation, reward, done, {}

    def apply_action_bitouch(
        self,
        motor_commands,
        velocity_action_repeat = 1,
        control_mode="tcp_velocity_control",
    ):
        # set the simulation with desired control points
        if control_mode == "tcp_velocity_control":
            self.embodiment_0.arm.set_target_tcp_velocities(motor_commands[0])
            self.embodiment_1.arm.set_target_tcp_velocities(motor_commands[1])
        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))
        self.apply_step_sim_velocity_move(velocity_action_repeat)

    def apply_step_sim_velocity_move(self, velocity_action_repeat = 1):
        for i in range(velocity_action_repeat):
            self.embodiment_0.arm.apply_gravity_compensation()
            self.embodiment_1.arm.apply_gravity_compensation()
            self._pb.stepSimulation()


    def get_two_robots_current_states(self):
        # Robot_0 pose in world and work frames.
        self.cur_tcp_pose_worldframe_robot_0 = self.embodiment_0.arm.get_tcp_pose()
        self.cur_tcp_pose_workframe_robot_0 = self.worldframe_to_workframe(self.cur_tcp_pose_worldframe_robot_0)
        
        (
            self.cur_tcp_pos_worldframe_robot_0 ,
            self.cur_tcp_rpy_worldframe_robot_0,
            self.cur_tcp_orn_worldframe_robot_0,
        ) = self.get_pos_rpy_orn_from_pose(self.cur_tcp_pose_worldframe_robot_0)
        (
            self.cur_tcp_pos_workframe_robot_0 ,
            self.cur_tcp_rpy_workframe_robot_0,
            self.cur_tcp_orn_workframe_robot_0,
        ) = self.get_pos_rpy_orn_from_pose(self.cur_tcp_pose_workframe_robot_0)

        # Robot_0 twist in world and work frames.
        self.cur_tcp_vel_worldframe_robot_0 = self.embodiment_0.arm.get_tcp_vel()
        self.cur_tcp_vel_workframe_robot_0 = self.worldvel_to_workvel(self.cur_tcp_vel_worldframe_robot_0)

        # Robot_1 pose in world and work frames.
        self.cur_tcp_pose_worldframe_robot_1 = self.embodiment_1.arm.get_tcp_pose()
        self.cur_tcp_pose_workframe_robot_1 = self.worldframe_to_workframe(self.cur_tcp_pose_worldframe_robot_1)
        
        (
            self.cur_tcp_pos_worldframe_robot_1 ,
            self.cur_tcp_rpy_worldframe_robot_1,
            self.cur_tcp_orn_worldframe_robot_1,
        ) = self.get_pos_rpy_orn_from_pose(self.cur_tcp_pose_worldframe_robot_1)
        (
            self.cur_tcp_pos_workframe_robot_1 ,
            self.cur_tcp_rpy_workframe_robot_1,
            self.cur_tcp_orn_workframe_robot_1,
        ) = self.get_pos_rpy_orn_from_pose(self.cur_tcp_pose_workframe_robot_1)

        # Robot_1 twist in world and work frames.
        self.cur_tcp_vel_worldframe_robot_1 = self.embodiment_0.arm.get_tcp_vel()
        self.cur_tcp_vel_workframe_robot_1 = self.worldvel_to_workvel(self.cur_tcp_vel_worldframe_robot_1)

    def get_obj_current_state(self):
        self.cur_obj_pose_worldframe = self.get_obj_pose_worldframe()
        (self.cur_obj_pos_workframe, 
         self.cur_obj_rpy_workframe, 
         self.cur_obj_orn_workframe
        ) = self.get_obj_pos_rpy_orn_workframe()

        (self.cur_obj_pos_worldframe, 
         self.cur_obj_rpy_worldframe, 
         self.cur_obj_orn_worldframe
        ) = self.get_obj_pos_rpy_orn_worldframe()

        (
            self.cur_obj_lin_vel_workframe,
            self.cur_obj_ang_vel_workframe,
        ) = self.get_obj_vel_workframe()

    def get_pos_rpy_orn_from_pose(self, pose):
        pos =  pose[:3]
        rpy =  pose[3:]
        orn = self._pb.getQuaternionFromEuler(rpy)
        return np.array(pos), np.array(rpy), np.array(orn)
    
    def xy_obj_0_dist_to_obj_1(self):
        """
        xy L2 distance from the current obj position to the goal.
        """
        dist = np.linalg.norm(
            self.cur_obj_pos_worldframe_0[:2] - self.cur_obj_pos_worldframe_1[:2]
        )
        # print(dist)
        return dist
    
    def worldframe_to_workframe(self, pose):
        return transform_eul(pose, self._workframe)

    def workframe_to_worldframe(self, pose):
        return inv_transform_eul(pose, self._workframe)

    def worldvel_to_workvel(self, vel):
        work_linvel = transform_vec_eul(vel[:3], self._workframe)
        work_angvel = transform_vec_eul(vel[3:], self._workframe)
        return np.array([*work_linvel, *work_angvel])

    def workvel_to_worldvel(self, vel):
        world_linvel = inv_transform_vec_eul(vel[:3], self._workframe)
        world_angvel = inv_transform_vec_eul(vel[3:], self._workframe)
        return np.array([*world_linvel, *world_angvel])

    def worldvec_to_workvec(self, vec):
        work_vec = transform_vec_eul(vec, self._workframe)
        return np.array(work_vec)

    def workvec_to_worldvec(self, vec):
        world_vec = inv_transform_vec_eul(vec, self._workframe)
        return np.array(world_vec)

    def worldframe_to_tcpframe(self, pose):
        cur_tcp_frame = self.embodiment.arm.get_tcp_pose()
        return transform_eul(pose, cur_tcp_frame)

    def tcpframe_to_worldframe(self, pose):
        cur_tcp_frame = self.embodiment.arm.get_tcp_pose()
        return inv_transform_eul(pose, cur_tcp_frame)

    def worldvel_to_tcpvel(self, vel):
        cur_tcp_frame = self.embodiment.arm.get_tcp_pose()
        tcp = transform_vec_eul(vel[:3], cur_tcp_frame)
        tcp_angvel = transform_vec_eul(vel[3:], cur_tcp_frame)
        return np.array([*tcp, *tcp_angvel])

    def tcpvel_to_worldvel(self, vel):
        cur_tcp_frame = self.embodiment.arm.get_tcp_pose()
        world_linvel = inv_transform_vec_eul(vel[:3], cur_tcp_frame)
        world_angvel = inv_transform_vec_eul(vel[3:], cur_tcp_frame)
        return np.array([*world_linvel, *world_angvel])

    def check_TCP_pos_lims(self, pose):
        """
        Clip the pose at the TCP limits specified.
        """
        return np.clip(pose, self._tcp_lims[:, 0], self._tcp_lims[:, 1])

    def check_TCP_vel_lims(self, embodiment, vels):
        """
        check whether action will take TCP outside of limits,
        zero any velocities that will.
        """
        if embodiment.robot_lv == "main_robot":
            cur_tcp_pos = self.cur_tcp_pos_workframe_robot_0
            cur_tcp_rpy = self.cur_tcp_rpy_workframe_robot_0
            tcp_lims = self._tcp_lims[0]
        elif embodiment.robot_lv == "slave_robot":
            cur_tcp_pos = self.cur_tcp_pos_workframe_robot_1
            cur_tcp_rpy = self.cur_tcp_rpy_workframe_robot_1
            tcp_lims = self._tcp_lims[1]
        # the direction that's exceeded
        if embodiment.update_init_rpy[2]==np.pi and embodiment.robot_lv=="slave_robot" :
            exceed_pos_llims = np.logical_and(
                cur_tcp_pos < tcp_lims[:3, 0], vels[:3] < 0
            )
            exceed_pos_ulims = np.logical_and(
                cur_tcp_pos > tcp_lims[:3, 1], vels[:3] > 0
            )
            exceed_rpy_llims = np.logical_and(
                cur_tcp_rpy[0:2] < tcp_lims[3:5, 0], vels[3:5] < 0
            )
            exceed_rpy_ulims = np.logical_and(
                cur_tcp_rpy[0:2] > tcp_lims[3:5, 1], vels[3:5] > 0
            )
            if cur_tcp_rpy[2] > 0:
                exceed_rpy_llims = np.insert(exceed_rpy_llims, 2, np.logical_and(
                    cur_tcp_rpy[2]-2*np.pi < tcp_lims[5, 0], vels[5] < 0
                ))
                # no limit here, just make up the number to match.
                exceed_rpy_ulims = np.insert(exceed_rpy_ulims, 2, np.logical_and(
                    cur_tcp_rpy[2]-2*np.pi < tcp_lims[5, 0], vels[5] < 0
                ))
            else:
                # no limit here, just make up the number to match.
                exceed_rpy_ulims = np.insert(exceed_rpy_ulims, 2 ,np.logical_and(
                    cur_tcp_rpy[2] > tcp_lims[5, 1], vels[5] > 0
                ))
                exceed_rpy_llims = np.insert(exceed_rpy_llims, 2, np.logical_and(
                    cur_tcp_rpy[2] < tcp_lims[5, 0], vels[5] < 0
                ))
        else:        
            exceed_pos_llims = np.logical_and(
                cur_tcp_pos < tcp_lims[:3, 0], vels[:3] < 0
            )
            exceed_pos_ulims = np.logical_and(
                cur_tcp_pos > tcp_lims[:3, 1], vels[:3] > 0
            )
            exceed_rpy_llims = np.logical_and(
                cur_tcp_rpy < tcp_lims[3:, 0], vels[3:] < 0
            )
            exceed_rpy_ulims = np.logical_and(
                cur_tcp_rpy > tcp_lims[3:, 1], vels[3:] > 0
            )
        # combine all bool arrays into one
        exceeded_pos = np.logical_or(exceed_pos_llims, exceed_pos_ulims)
        exceeded_rpy = np.logical_or(exceed_rpy_llims, exceed_rpy_ulims)
        exceeded = np.concatenate([exceeded_pos, exceeded_rpy])

        # cap the velocities at 0 if limits are exceeded
        capped_vels = np.array(vels)
        capped_vels[np.array(exceeded)] = 0

        # transform for mg400 from robot to world
        if embodiment.robot_lv=="main_robot":
            y = capped_vels[0]
            x = -capped_vels[1] 
        else:
            y = -capped_vels[0]
            x = capped_vels[1] 
        capped_vels[0] = x
        capped_vels[1] = y

        return capped_vels

    def get_extended_feature_array(self):
        """
        Get feature to extend current observations.
        """
        return np.array([])

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel.
        """

        tactile_obs_0 = self.embodiment_0.get_tactile_observation()
        tactile_obs_1 = self.embodiment_1.get_tactile_observation()
        observation = np.concatenate([tactile_obs_0, tactile_obs_1], axis=1)
        observation = observation[..., np.newaxis]
        return observation
    
    def get_visual_obs(self):
        """
        Returns the rgb image from a static environment camera.
        """
        visual_rgba, _, _ = self.embodiment_0.get_visual_observation()
        return visual_rgba[..., :3]

    def get_observation(self):
        """
        Returns the observation dependent on which mode is set.
        """

        observation_mode = self._env_params["observation_mode"]
        
        # check correct obs type set
        if observation_mode not in [
            "oracle",
            "tactile",
            "visual",
            "visuotactile",
            "tactile_and_feature",
            "visual_and_feature",
            "visuotactile_and_feature",
        ]:
            sys.exit("Incorrect observation mode specified: {}".format(observation_mode))

        observation = {}
        # use direct pose info to check if things are working
        if "oracle" in observation_mode:
            observation["oracle"] = self.get_oracle_obs()

        # observation is just the tactile sensor image
        if "tactile" in observation_mode:
            observation["tactile"] = self.get_tactile_obs()

        # observation is rgb environment camera image
        if any(obs in observation_mode for obs in ["visual", "visuo"]):
            observation["visual"] = self.get_visual_obs()

        # observation is mix image + features (pretending to be image shape)
        if "feature" in observation_mode:
            observation["extended_feature"] = self.get_extended_feature_array()

        return observation

    def render(self, mode="rgb_array"):
        """
        Most rendering handled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # resize tactile to match rgb if rendering in higher res
        if self._tactile_sensor_params["image_size"] != self._visual_sensor_params["image_size"]:
            tactile_array = cv2.resize(tactile_array, tuple(self._visual_sensor_params["image_size"]))

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        # setup plot for rendering
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = "render_window_{}".format(self._seed)
            else:
                self.window_name = "render_window"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array

    def draw_workframe(self, lifetime=0.1):
        draw_frame(self._workframe, lifetime=lifetime)

    def draw_tcp_lims(self, lifetime=0.1):
        draw_box(self._workframe, self._tcp_lims)
