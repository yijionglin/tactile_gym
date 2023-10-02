import os, sys
import numpy as np
from tactile_gym.assets import add_assets_path, add_data_save_path
from tactile_gym.envs.bitouch.bipush_env.poses import (
    rest_poses_dict,
    EEs_poses_sets,
)
from tactile_gym.envs.bitouch.base_bitouch_object_env import BaseBitouchObjectEnv
import cv2
from tactile_sim.utils.pybullet_draw_utils import draw_vector
from opensimplex import OpenSimplex

from ipdb import set_trace

class BipushEnv(BaseBitouchObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},

    ):
        # env specific values
        self.env_name = "bipush"
        self.termination_pos_dist = 0.025
        self.embed_dist = 0.005
        self.obj_width = 0.40
        self.obj_length = 0.06
        self.obj_height = 0.06
        self.visualise_goal = False
        self.if_goal_visiualization = True
        self.basic_init_object_goal_offset_x = 0.01 + EEs_poses_sets[robot_arm_params["type"]]["main_robot"]["update_init_pos"][0]
        
        self.basic_init_object_goal_offset_y = 0.00
        self.save_traj_flag = False

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.0, 0.0, 0.1 + self.obj_height/2, -np.pi, 0.0, 0.0])
        TCP_lims_robot_0 = np.zeros(shape=(6, 2))
        TCP_lims_robot_0[0, 0], TCP_lims_robot_0[0, 1] = -0.25, 0.10  # x lims
        TCP_lims_robot_0[1, 0], TCP_lims_robot_0[1, 1] = 0.10, 0.20  # y lims
        TCP_lims_robot_0[2, 0], TCP_lims_robot_0[2, 1] = -0.0, 0.0  # z lims
        TCP_lims_robot_0[3, 0], TCP_lims_robot_0[3, 1] = -0.0, 0.0  # roll lims
        TCP_lims_robot_0[4, 0], TCP_lims_robot_0[4, 1] = -0.0, 0.0  # pitch lims
        TCP_lims_robot_0[5, 0], TCP_lims_robot_0[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims
        TCP_lims_robot_1 = TCP_lims_robot_0.copy()
        TCP_lims_robot_1[1,:] -= 0.3
        TCP_lims_list = [TCP_lims_robot_0, TCP_lims_robot_1]
        env_params["tcp_lims"] = TCP_lims_list
        self.rand_obj_mass = env_params["rand_obj_mass"]
        self.traj_type = env_params["traj_type"]
        self.rand_init_orn = env_params["rand_init_orn"]
        
        # add environment specific robot arm parameters
        
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]][tactile_sensor_params["type"]]
        robot_arm_params["base_pos_and_init_pos"] = EEs_poses_sets[robot_arm_params["type"]]
        robot_arm_params["base_rpy_and_init_rpy"] = EEs_poses_sets[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"
        
        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "fixed"
        tactile_sensor_params["dynamics"] =  {'stiffness': 50, 'damping': 120, 'friction':10.0}
        tactile_sensor_params["random_friction"] = False
        tactile_sensor_params["random_damping"] = False
        
        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 1.5
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -60
        visual_sensor_params["pos"] = [-0.42, 0.0, -0.65]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        a_dim = int(len(robot_arm_params['control_dofs'])/2)
        self.movement_mode = ''.join(robot_arm_params['control_dofs'][:a_dim])
        super(BipushEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)
        

    def setup_rgb_obs_camera_params(self):
        self.rgb_cam_pos = [-0.55, 0.0, -0.85]
        self.rgb_cam_dist = 1.5
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -60
        # self.rgb_image_size = self._image_size
        self.rgb_image_size = [2048,2048]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def setup_object(self):
        """
        Set vars for loading an object.
        Just used in initialzing an env lass.
        """
        self.object_init_offset_length_y = 0.0
        self.object_init_offset_length_x = self.embodiment_0.update_init_pose[0] # can be used for embedding
        self.init_obj_pos = [self._workframe[0] + self.object_init_offset_length_x + self.obj_height / 2, self.object_init_offset_length_y + self._workframe[1], self._workframe[2]]
        # The object initial orientation needs to be consistent with the work frame
        self.init_obj_rpy = self._workframe[3:]
        self.init_obj_orn = self._pb.getQuaternionFromEuler(self.init_obj_rpy)
        # get paths
        self.object_path = add_assets_path("bitouch/bipush_obj/short_cube.urdf")
        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """
        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()
        self.reset_counter += 1
        self._env_step_counter = 0
        # update the workframe to a new position if randomisations are on
        self.reset_task()
        # make room for the object
        reset_tcp_pose_0_worldframe = self.workframe_to_worldframe(np.array([*[-0.25,0.14,0], *self.embodiment_0.update_init_rpy]))
        reset_tcp_pose_1_worldframe = self.workframe_to_worldframe(np.array([*[-0.25,-0.14,0], *self.embodiment_1.update_init_rpy]))
        self.embodiment_0.reset(reset_tcp_pose_0_worldframe)
        self.embodiment_1.reset(reset_tcp_pose_1_worldframe)
        self.reset_object()
        self.make_goal()
        self.get_step_data()
        self._observation = self.get_observation()
        return self._observation

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        if self.rand_init_orn:
            self.init_obj_ang = self.np_random.uniform(-np.pi / 32, np.pi / 32)
        else:
            self.init_obj_ang = 0.0

        self.init_obj_orn = self._pb.getQuaternionFromEuler(
            [self.init_obj_rpy[0], self.init_obj_rpy[1], self.init_obj_rpy[2] + self.init_obj_ang]
        )
        self._pb.resetBasePositionAndOrientation(
            self.obj_id, self.init_obj_pos, self.init_obj_orn
        )
        # perform object dynamics randomisations
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=0.065,
            spinningFriction=0.00,
            rollingFriction=0.00,
            restitution=0.0,
            frictionAnchor=1,
            collisionMargin=0.0001,
        )

        if self.rand_obj_mass:
            obj_mass = self.np_random.uniform(0.4, 0.8)
            self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)

    def full_reset(self):
        """
        Pybullet can encounter some silent bugs, particularly when unloading and
        reloading objects. This will do a full reset every once in a while to
        clear caches.
        """
        self._pb.resetSimulation()
        self.load_environment()
        self.load_object(self.visualise_goal)
        self.embodiment_0.full_reset()
        self.embodiment_1.full_reset()
        self.reset_counter = 0


    def load_trajectory(self):

        # relatively easy traj
        self.traj_n_points = 12
        self.traj_spacing = 0.025
        self.traj_max_perturb = 0.08
        # place goals at each point along traj
        self.traj_ids = []
        for i in range(int(self.traj_n_points)):
            # just loading goals here, no placement yet
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
            # changeVisualShape(objectUniqueId, linkIndex, rgbaColor)
            self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
            self.traj_ids.append(traj_point_id)

    def update_trajectory(self):

        # setup traj arrays
        self.targ_traj_list_id = -1
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_orn_workframe = np.zeros(shape=(self.traj_n_points, 4))

        if self.traj_type == "simplex":
            self.update_trajectory_simplex()
        elif self.traj_type == "straight":
            self.update_trajectory_straight()
        elif self.traj_type == "sin":
            self.update_trajectory_sin()
        else:
            sys.exit("Incorrect traj_type specified: {}".format(self.traj_type))

        # calc orientation to place object at
        self.traj_rpy_workframe[:, 2] = np.gradient(
            self.traj_pos_workframe[:, 1], self.traj_spacing
        )
        for i in range(int(self.traj_n_points)):
            # get workframe orn
            self.traj_orn_workframe[i] = self._pb.getQuaternionFromEuler(
                self.traj_rpy_workframe[i]
            )
            # convert worldframe
            pose_worldframe = self.workframe_to_worldframe(
                np.array([*self.traj_pos_workframe[i], *self.traj_rpy_workframe[i]])
            )
            pos_worldframe, rpy_worldframe, orn_worldframe = self.get_pos_rpy_orn_from_pose(pose_worldframe)
            # place goal
            self._pb.resetBasePositionAndOrientation(
                self.traj_ids[i], pos_worldframe, orn_worldframe
            )
            self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0.5])
        traj_idx = int(self.reset_counter / 2)
        if self.save_traj_flag:
            pos_save_path = add_data_save_path('sb3_helpers/collected_data/traj_pos_{}.npy'.format(traj_idx))
            rpy_save_path = add_data_save_path('sb3_helpers/collected_data/traj_rpy_{}.npy'.format(traj_idx))
            np.save(pos_save_path, self.traj_pos_workframe)
            np.save(rpy_save_path, self.traj_rpy_workframe)

    def update_trajectory_simplex(self):
        """
        Generates smooth trajectory of goals
        """
        # initialise noise
        try:
            seed=self.np_random.randint(1e8)
        except AttributeError as e:
            print(f"AttributeError: {e}")
            print("Using self.np_random.integers instead.")
            seed=self.np_random.integers(1e8)
        simplex_noise = OpenSimplex(seed=seed)
        init_offset_x = self.basic_init_object_goal_offset_x + self.obj_height / 2 + self.traj_spacing
        init_offset_y = self.basic_init_object_goal_offset_y
        # generate smooth 1d traj using opensimplex
        first_run = True
        for i in range(int(self.traj_n_points)):
            noise = simplex_noise.noise2d(x=i * 0.1, y=1) * self.traj_max_perturb
            if first_run:
                init_offset_y -= noise
                first_run = False

            x = init_offset_x + (i * self.traj_spacing)
            y = init_offset_y + noise
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def update_trajectory_straight(self):
        """
        Generates smooth trajectory of goals
        """
        traj_ang = self.np_random.uniform(-np.pi / 12, np.pi / 12)
        traj_ang = np.pi / 12
        init_offset_x = self.basic_init_object_goal_offset_x + self.obj_height / 2 + self.traj_spacing
        init_offset_y = self.basic_init_object_goal_offset_y
        for i in range(int(self.traj_n_points)):

            dir_x = np.cos(traj_ang)
            dir_y = np.sin(traj_ang)
            dist = i * self.traj_spacing

            x = init_offset_x + dist * dir_x
            y = init_offset_y + dist * dir_y
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def update_trajectory_sin(self):
        """
        Generates smooth trajectory of goals
        """
        traj_idx = int(self.reset_counter / 2)
        curve_dir = -1 if traj_idx % 2 == 0 else +1

        init_offset_x = self.basic_init_object_goal_offset_x + self.obj_height / 2 + self.traj_spacing
        init_offset_y = self.basic_init_object_goal_offset_y

        def curve_func(x):
            y = curve_dir*0.025*np.sin(20*(x-init_offset_x))
            return y

        for i in range(int(self.traj_n_points)):
            dist = (i*self.traj_spacing)
            x = init_offset_x + dist
            y = -curve_func(x)
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """
        # update the curren trajecory
        self.update_trajectory()

        # set goal as first point along trajectory
        self.update_goal()

    def update_goal(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id += 1

        if self.targ_traj_list_id >= self.traj_n_points:
            # if achieved all goals, return False to terminate this episode
            return False
        else:
            # get the next goal pb id
            self.goal_id = self.traj_ids[self.targ_traj_list_id]

            # get goal pose in world frame
            (
                self.goal_pos_worldframe, # used to calc xyz_obj_dist_to_goal
                self.goal_orn_worldframe, # used to calc orn_obj_dist_to_goal
            ) = self._pb.getBasePositionAndOrientation(self.goal_id)
            
            # goal_rpy_worldframe is not used in pushing task
            self.goal_rpy_worldframe = self._pb.getEulerFromQuaternion(
                self.goal_orn_worldframe
            )

            # create variables for goal pose in workframe to use later
            self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
            self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id]
            self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id] 

            # change colour of new target goal
            self._pb.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached, which is in red.
            prev_goal_traj_list_id = (
                self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None
            )
            if prev_goal_traj_list_id is not None:
                self._pb.changeVisualShape(
                    self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5]
                )

            return True

    def encode_TCP_frame_actions(self, actions, robot):
        encoded_actions = np.zeros(6)
        if robot.robot_lv == 'main_robot':
            tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_0) 
        elif robot.robot_lv == 'slave_robot':
            tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_1) 
        else:
            assert "Error: no robot is found."
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        par_vector = np.array([1, 0, 0])  # outwards from tip
        perp_vector = np.array([0, -1, 0])  # perp to tip
        par_tip_direction = tip_rot_matrix.dot(par_vector)
        perp_tip_direction = tip_rot_matrix.dot(perp_vector)

        workframe_par_tip_direction = self.worldvec_to_workvec(
            par_tip_direction
        )
        workframe_perp_tip_direction = self.worldvec_to_workvec(
            perp_tip_direction
        )

        if self.movement_mode == "TyRz":
            # translate the direction
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)
            # auto move in the dir tip is pointing
            par_scale = 1.0 * self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TxRz":
            # translate the direction
            perp_scale = 0
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)
            # auto move in the dir tip is pointing
            par_scale = 1.2 * self.max_action + actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TxTyRz":
            # translate the direction
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = 1.0 * self.max_action - actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[2]

        else:
            assert "Error: input movement mode has not been defined"
        return encoded_actions
        
    def encode_work_frame_actions(self, actions, robot):
        """
        Return actions as np.array in correct places for sending to robot arm.
        """

        encoded_actions = np.zeros(6)
        
        if self.movement_mode == "y":
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]

        if self.movement_mode == "yRz":
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]
            encoded_actions[5] = actions[1]

        elif self.movement_mode == "xyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[5] = actions[2]

        return encoded_actions

    def encode_actions(self, actions, robot):
        # scale and embed actions appropriately
        if self.movement_mode in ["y", "yRz", "xyRz"]:
            assert "Error: this movement mode is not defined yet."
            encoded_actions = self.encode_work_frame_actions(actions)
        elif self.movement_mode in ["TyRz","TxRz", "TxTyRz"]:
            encoded_actions = self.encode_TCP_frame_actions(actions, robot)
        return encoded_actions

    def get_pos_rpy_orn_from_pose(self, pose):
        pos =  pose[:3]
        rpy =  pose[3:]
        orn = self._pb.getQuaternionFromEuler(rpy)
        return np.array(pos), np.array(rpy), np.array(orn)

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel.
        """
        # get image from sensor
        tactile_obs_1 = self.embodiment_0.get_tactile_observation()
        tactile_obs_2 = self.embodiment_1.get_tactile_observation()
        observation = np.concatenate([tactile_obs_1, tactile_obs_2], axis=1)
        observation = observation[..., np.newaxis]
        return observation
    
    def get_step_data(self):
        # For computing the reward as well as for  the observation.
        # get state of tcp and obj per step
        self.get_two_robots_current_states()
        self.get_obj_current_state()
        reward = self.get_reward()
        # get rl info
        done = self.termination()
        return reward, done

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        tip_obj_orn_dist = self.cos_tcp_dist_to_obj()

        # weights for rewards
        W_obj_goal_pos = 2.0
        W_obj_goal_orn = 1.0
        W_tip_obj_orn = 1.0

        penalty = (self.traj_n_points - self.targ_traj_list_id) * (self.traj_n_points - self.targ_traj_list_id)
        # sum rewards with multiplicative factors
        reward = -(
            (W_obj_goal_pos * obj_goal_pos_dist)
            + (W_obj_goal_orn * obj_goal_orn_dist)
            + (W_tip_obj_orn * tip_obj_orn_dist)
        ) * penalty

        return reward

    def cos_tcp_dist_to_obj(self):
        """
        Cos distance from current orientation of the TCP to the current
        orientation of the object

        dont come here with MARL at this moment.
        """
        # get normal vector of object
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        obj_init_vector = np.array([1, 0, 0])
        obj_vector = obj_rot_matrix.dot(obj_init_vector)
        # get vector of tactip tip, directed through tip body
        tip_rot_matrix_robot = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_0)
        tip_rot_matrix_robot = np.array(tip_rot_matrix_robot).reshape(3, 3)
        tip_init_vector_robot = np.array([1, 0, 0])
        tip_vector_robot = tip_rot_matrix_robot.dot(tip_init_vector_robot)
        # get the cosine similarity/distance between the two vectors
        cos_sim_robot = np.dot(obj_vector, tip_vector_robot) / (
            np.linalg.norm(obj_vector) * np.linalg.norm(tip_vector_robot)
        )
        cos_dist_robot = 1 - cos_sim_robot
        # get vector of tactip tip, directed through tip body
        tip_rot_matrix_robot_2 = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_1)
        tip_rot_matrix_robot_2 = np.array(tip_rot_matrix_robot_2).reshape(3, 3)
        tip_init_vector_robot_2 = np.array([1, 0, 0])
        tip_vector_robot_2 = tip_rot_matrix_robot_2.dot(tip_init_vector_robot_2)
        # get the cosine similarity/distance between the two vectors
        cos_sim_robot_2 = np.dot(obj_vector, tip_vector_robot_2) / (
            np.linalg.norm(obj_vector) * np.linalg.norm(tip_vector_robot_2)
        )
        cos_dist_robot_2 = 1 - cos_sim_robot_2
        cos_dist_robots = (cos_dist_robot + cos_dist_robot_2)/2

        return cos_dist_robots
    
    def termination(self):
        """
        Criteria for terminating an episode.
        """
        # check if near goal, change the goal if so
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:
            # update the goal (if not at end of traj)
            goal_updated = self.update_goal()
            # if self.targ_traj_list_id > self.traj_n_points-1:
            if not goal_updated:
                return True
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on object
        # stack into array
        observation = np.hstack(
            [
                *self.cur_tcp_pose_workframe_robot_0,
                *self.cur_tcp_vel_workframe_robot_0,

                *self.cur_tcp_pose_workframe_robot_1,
                *self.cur_tcp_vel_workframe_robot_1,

                *self.cur_obj_pos_workframe,
                *self.cur_obj_rpy_workframe,

                *self.cur_obj_lin_vel_workframe,
                *self.cur_obj_ang_vel_workframe,
            ]
        )


        return observation

    def get_extended_feature_array(self):

        feature_array = np.array(
            [
                *self.cur_tcp_pose_workframe_robot_0,
                *self.cur_tcp_pose_workframe_robot_1,
                *self.goal_pos_workframe,
                *self.goal_rpy_workframe,
            ]
        )
        return feature_array


    def task_callback(self):
        if isinstance(self._env_step_counter%50, int) == True:
            self.object

    def render(self, mode="rgb_array"):
        """
        Most rendering handeled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()
        tactile_processed_images_list = []
        # get the current tactile images and reformat to match rgb array
        tactile_array_1 = self.embodiment_0.get_tactile_observation()
        tactile_array_2 = self.embodiment_1.get_tactile_observation()
        tactile_arrays_list = [tactile_array_1, tactile_array_2]

        for tactile_array in tactile_arrays_list:
            tactile_array = tactile_array[..., np.newaxis]
            tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)
            tactile_array = cv2.resize(tactile_array, tuple([1024,1024]))
            tactile_processed_images_list.append(tactile_array)
        # add goal indicator in approximate position
        render_array = np.concatenate([ tactile_processed_images_list[0], \
                                                tactile_processed_images_list[1]], axis=1)
        render_array = np.concatenate([rgb_array, render_array], axis=0)
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = 'render_window_{}'.format(self._seed)
            else:
                self.window_name = 'render_window'
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array

    def draw_obj_workframe(self):
        self._pb.changeVisualShape(
          self.obj_id  , -1, rgbaColor=[1, 1, 1, 0.3]
        )
        return super().draw_obj_workframe()
