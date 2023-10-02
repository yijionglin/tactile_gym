import os, sys
import numpy as np
from tactile_gym.assets import add_assets_path
from tactile_gym.envs.bitouch.bigather_env.poses import (
    rest_poses_dict,
    EEs_poses_sets,
)
from tactile_gym.envs.bitouch.base_bitouch_object_env import BaseBitouchObjectEnv
import cv2
from tactile_sim.utils.pybullet_draw_utils import draw_vector
from ipdb import set_trace

class BigatherEnv(BaseBitouchObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},

    ):
        # env specific values
        self.env_name = "bigather"
        self.termination_pos_dist = 0.08
        self.goal_update_pos_dist = 0.03
        self.embed_dist = 0.005
        self.obj_width = 0.06
        self.obj_length = 0.06
        self.obj_height = 0.1
        self.visualise_goal = False
        self.if_goal_visiualization = True
        self.if_use_obj_xy_info = 0
        self.if_use_obj_Rz_info = 0
        self.if_apply_force_perturbation = True
        
        # add environment specific env parameters
        env_params["workframe"] = np.array([0.0, 0.0, 0.1 + self.obj_height/2, -np.pi, 0.0, -np.pi])
        TCP_lims_robot_0 = np.zeros(shape=(6, 2))
        TCP_lims_robot_0[0, 0], TCP_lims_robot_0[0, 1] = -0.30, 0.10  # x lims
        TCP_lims_robot_0[1, 0], TCP_lims_robot_0[1, 1] = -0.30, 0.30  # y lims
        TCP_lims_robot_0[2, 0], TCP_lims_robot_0[2, 1] = -0.12, 0.12  # z lims
        TCP_lims_robot_0[3, 0], TCP_lims_robot_0[3, 1] = -100.0, 0.0  # roll lims
        TCP_lims_robot_0[4, 0], TCP_lims_robot_0[4, 1] = -0.0, 0.0  # pitch lims
        TCP_lims_robot_0[5, 0], TCP_lims_robot_0[5, 1] = -100 * np.pi / 180, 100 * np.pi / 180  # yaw lims
        TCP_lims_robot_1 = TCP_lims_robot_0.copy()
        TCP_lims_robot_1[0, 0], TCP_lims_robot_1[0, 1] = -0.10, 0.30  # x lims
        TCP_lims_robot_1[5, 0], TCP_lims_robot_1[5, 1] = (-100-180) * np.pi / 180, (100-180) * np.pi / 180  # yaw lims
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
        tactile_sensor_params["dynamics"] =  {'stiffness': 45, 'damping': 100, 'friction':10.0}
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
        super(BigatherEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)


    def load_goal_line(self):
        self.goal_line_id = self._pb.loadURDF(
            self.goal_line_path, self.init_obj_pos, self.init_obj_orn, useFixedBase=True
        )
        self._pb.setCollisionFilterGroupMask(self.goal_line_id, -1, 0, 0)

        

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
        Set vars for loading two objects.
        Just used in initialzing an env lass.
        """
        # The init pose of object is defined in world space
        # define an initial position for the objects (world coords)
        self.init_obj_pos_0 = [-0.19, 0.01 , self._workframe[2]]
        self.init_obj_pos_1 = [0.19, 0.01 , self._workframe[2]]
        # The object initial orientation needs to be consistent with the work frame
        self.init_obj_rpy_0 = self._workframe[3:]
        self.init_obj_rpy_1 = self._workframe[3:]
        self.init_obj_orn_0 = self._pb.getQuaternionFromEuler(self.init_obj_rpy_0)
        self.init_obj_orn_1 = self._pb.getQuaternionFromEuler(self.init_obj_rpy_1)
        # get paths
        self.object_path = add_assets_path("bitouch/bigather_obj/cube.urdf")
        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")


    def load_object(self, visualise_goal=True):
        """
        Load an object that is used
        """
        # load temp object and goal indicators so they can be more conveniently updated
        self.obj_id_0 = self._pb.loadURDF(
            self.object_path, self.init_obj_pos_0, self.init_obj_orn_0
        )
        self.obj_id_1 = self._pb.loadURDF(
            self.object_path, self.init_obj_pos_1, self.init_obj_orn_1
        )
        if visualise_goal:
            self.goal_indicator = self._pb.loadURDF(
                self.goal_path, self.init_obj_pos_0, [0, 0, 0, 1], useFixedBase=True
            )
            self._pb.changeVisualShape(
                self.goal_indicator, -1, rgbaColor=[1, 0, 0, 0.5]
            )
            self._pb.setCollisionFilterGroupMask(self.goal_indicator, -1, 0, 0)
            
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
        self.update_objs_robots_init_poses()
        # make room for the object
        reset_tcp_pose_0_worldframe = self.workframe_to_worldframe(np.array([*self.reset_TCP_pos_embodiment_0, *self.embodiment_0.update_init_rpy]))
        reset_tcp_pose_1_worldframe = self.workframe_to_worldframe(np.array([*self.reset_TCP_pos_embodiment_1, *self.embodiment_1.update_init_rpy]))
        self.embodiment_0.reset(reset_tcp_pose_0_worldframe)
        self.embodiment_1.reset(reset_tcp_pose_1_worldframe)
        self.reset_object()
        self.make_goal()
        self.get_step_data()
        self._observation = self.get_observation()
        return self._observation

    def update_objs_robots_init_poses(self):
        # because the arm cannot reach too far in the very beginning so we fine tune the init pos of the objs
        init_mode_for_obj_0 = self.np_random.choice([0, 1, 2, 2 , 2 ,2, 3])
        if init_mode_for_obj_0 == 0:
            self.init_obj_pos_0[1] = -self.np_random.uniform(0, 0.12)
            self.init_obj_pos_0[0] = self.np_random.uniform(0.15, (0.25-0.20)/(0.12-0.00)*-self.init_obj_pos_0[1] + 0.20  )
        elif init_mode_for_obj_0 == 1:
            self.init_obj_pos_0[1] = -self.np_random.uniform(-0.05, 0.0)
            self.init_obj_pos_0[0] = self.np_random.uniform(0.07, 0.12 )
        elif init_mode_for_obj_0 == 2: 
            self.init_obj_pos_0[1] = -self.np_random.uniform(0.0, 0.08)
            self.init_obj_pos_0[0] = self.np_random.uniform(0.15, 0.20 )
        else:
            self.init_obj_pos_0[1] = -self.np_random.uniform(-0.06, -0.04)
            self.init_obj_pos_0[0] = self.np_random.uniform(0.00, 0.03)

        if init_mode_for_obj_0 != 3 :
            init_mode_for_obj_1 = self.np_random.choice([0, 1, 2, 2 , 2 ,2, 3])
        else:
            init_mode_for_obj_1 = self.np_random.choice([0, 1, 2, 2 , 2 ,2])

        
        if init_mode_for_obj_1 == 0:
            self.init_obj_pos_1[1] = self.np_random.uniform(0, 0.12)
            self.init_obj_pos_1[0] = -self.np_random.uniform(0.15, (0.25-0.20)/(0.12-0.00)*-self.init_obj_pos_1[1] + 0.20 )
        elif init_mode_for_obj_1 == 1:
            self.init_obj_pos_1[1] = self.np_random.uniform(-0.05, 0.0)
            self.init_obj_pos_1[0] = -self.np_random.uniform(0.07, 0.12)
        elif init_mode_for_obj_1 == 2: 
            self.init_obj_pos_1[1] = self.np_random.uniform(0.0, 0.08)
            self.init_obj_pos_1[0] = -self.np_random.uniform(0.15, 0.20)
        else:
            self.init_obj_pos_1[1] = self.np_random.uniform(-0.06, -0.04)
            self.init_obj_pos_1[0] = -self.np_random.uniform(0.00, 0.03)

        # The tcp pos is in work frame, so it needs some transformation from obj pose.
        self.reset_TCP_pos_embodiment_0 = [-(self.init_obj_pos_0[0] + self.obj_length/2), self.init_obj_pos_0[1], 0 ]
        self.reset_TCP_pos_embodiment_1 =  [-(self.init_obj_pos_1[0] - self.obj_length/2), self.init_obj_pos_1[1], 0 ]

    def reset_object(self):
        """
        Reset the base poses of objects on reset,
        can also adjust physics params here.
        """
        self.if_use_obj_xy_info = self.np_random.choice([0, 1])
        self.if_use_obj_Rz_info = self.np_random.choice([0, 1])
        self.obj_id_0 = self._pb.loadURDF(
            self.object_path, self.init_obj_pos_0, self.init_obj_orn_0
        )
        self._pb.changeVisualShape(self.obj_id_0, -1, rgbaColor=[1, 0 , 0,1])

        self.obj_id_1 = self._pb.loadURDF(
            self.object_path, self.init_obj_pos_1, self.init_obj_orn_1
        )
        self._pb.changeVisualShape(self.obj_id_1, -1, rgbaColor=[0, 1, 0,1])

        for obj_id in [self.obj_id_0, self.obj_id_1]:
            self._pb.changeDynamics(
                obj_id,
                -1,
                lateralFriction=0.065,
                spinningFriction=0.00,
                rollingFriction=0.00,
                restitution=0.0,
                frictionAnchor=1,
                collisionMargin=0.0001,
            )

        obj_mass = 0.4
        self._pb.changeDynamics(self.obj_id_0, -1, mass=obj_mass)
        self._pb.changeDynamics(self.obj_id_1, -1, mass=obj_mass)
        if self.rand_obj_mass:
            obj_mass_0 = self.np_random.uniform(0.4, 0.8)
            obj_mass_1 = self.np_random.uniform(0.4, 0.8)
            self._pb.changeDynamics(self.obj_id_0, -1, mass=obj_mass_0)
            self._pb.changeDynamics(self.obj_id_1, -1, mass=obj_mass_1)


    def reset_task(self):
        self._pb.removeBody(self.obj_id_0)
        self._pb.removeBody(self.obj_id_1)

        self.if_apply_force_perturbation = self.np_random.choice([True, False, False])
        if self.if_apply_force_perturbation:
            self.per_direcction_for_obj_1 = self.np_random.choice([-1, 1, None])
            self.per_direcction_for_obj_0 = self.np_random.choice([-1, 1, None])
            self.stop_apply_force_pert_dist = self.np_random.uniform(0.18, 0.21)
            self.apply_pert_prob = self.np_random.uniform(0.005, 0.01)

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
        self.traj_n_points = 10
        self.traj_spacing = 0.025

        # place goals at each point along traj
        self.traj_ids = []
        for i in range(int(self.traj_n_points)):
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
            if not self.if_goal_visiualization:
                self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0])
            else:
                self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
            self.traj_ids.append(traj_point_id)

    def update_trajectory(self):

        # setup traj arrays
        self.targ_traj_list_id_for_obj_0 = -1
        self.targ_traj_list_id_for_obj_1 = 10
        self.if_goal = True
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_orn_workframe = np.zeros(shape=(self.traj_n_points, 4))



        if self.traj_type == "obj_straight_connection":
            self.update_trajectory_obj_straight_connection()
            if not self.if_goal:
                return True
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
            if not self.if_goal_visiualization:
                self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0])
            else:
                self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0.5])

    def update_trajectory_obj_straight_connection(self):

        self.get_two_objs_current_states()
        traj_ang = (self.cur_obj_pos_workframe_1[1] - self.cur_obj_pos_workframe_0[1]) / (self.cur_obj_pos_workframe_1[0] - self.cur_obj_pos_workframe_0[0])
        self.traj_spacing = np.linalg.norm(self.cur_obj_pos_workframe_1 - self.cur_obj_pos_workframe_0) / 10
        if np.abs(traj_ang) > 1 or (self.cur_obj_pos_workframe_0[0] - self.cur_obj_pos_workframe_1[0]) > 0:
            self.if_goal = False
            return True
            
        x_init_offset = self.cur_obj_pos_workframe_0[0]
        y_init_offset = self.cur_obj_pos_workframe_0[1]

        for i in range(int(self.traj_n_points)):

            dir_x = np.cos(traj_ang)
            dir_y = np.sin(traj_ang)
            dist = i * self.traj_spacing

            x = x_init_offset + dist * dir_x
            y = y_init_offset + dist * dir_y
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """
        # update the curren trajecory
        self.obj_connection_goal_update_counter = 0
        
        self.update_trajectory()
        if not self.if_goal:
            return True
        # set goal as first point along trajectory
        self.update_goal_for_obj_0()
        self.update_goal_for_obj_1()

    def update_goal_for_obj_1(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id_for_obj_1 -= 1
        if self.targ_traj_list_id_for_obj_1 <= 0:
            return False
        else:
            self.goal_id_for_obj_1 = self.traj_ids[self.targ_traj_list_id_for_obj_1]

            # get goal pose in world frame
            (
                self.goal_pos_worldframe_for_obj_1,
                self.goal_orn_worldframe_for_obj_1,
            ) = self._pb.getBasePositionAndOrientation(self.goal_id_for_obj_1)
            self.goal_rpy_worldframe_for_obj_1 = self._pb.getEulerFromQuaternion(
                self.goal_orn_worldframe_for_obj_1
            )

            # create variables for goal pose in workframe to use later
            self.goal_pos_workframe_for_obj_1 = self.traj_pos_workframe[self.targ_traj_list_id_for_obj_1]
            self.goal_orn_workframe_for_obj_1 = self.traj_orn_workframe[self.targ_traj_list_id_for_obj_1]
            self.goal_rpy_workframe_for_obj_1 = self.traj_rpy_workframe[self.targ_traj_list_id_for_obj_1]

            # change colour of new target goal
            if not self.if_goal_visiualization:
                self._pb.changeVisualShape(self.goal_id_for_obj_1, -1, rgbaColor=[0, 0, 1, 0])
            else:
                self._pb.changeVisualShape(self.goal_id_for_obj_1, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached
            prev_goal_traj_list_id = (
                self.targ_traj_list_id_for_obj_1 + 1 if self.targ_traj_list_id_for_obj_1 < 9 else None
            )
            if prev_goal_traj_list_id is not None:
                if not self.if_goal_visiualization:
                    self._pb.changeVisualShape(
                        self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0]
                    )
                else:
                    self._pb.changeVisualShape(
                        self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5]
                    )

            return True

    def update_goal_for_obj_0(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id_for_obj_0 += 1

        if self.targ_traj_list_id_for_obj_0 >= self.traj_n_points:
            return False
        else:
            self.goal_id_for_obj_0 = self.traj_ids[self.targ_traj_list_id_for_obj_0]

            # get goal pose in world frame
            (
                self.goal_pos_worldframe_for_obj_0,
                self.goal_orn_worldframe_for_obj_0,
            ) = self._pb.getBasePositionAndOrientation(self.goal_id_for_obj_0)
            self.goal_rpy_worldframe_for_obj_0 = self._pb.getEulerFromQuaternion(
                self.goal_orn_worldframe_for_obj_0
            )
            # create variables for goal pose in workframe to use later
            self.goal_pos_workframe_for_obj_0 = self.traj_pos_workframe[self.targ_traj_list_id_for_obj_0]
            self.goal_orn_workframe_for_obj_0 = self.traj_orn_workframe[self.targ_traj_list_id_for_obj_0]
            self.goal_rpy_workframe_for_obj_0 = self.traj_rpy_workframe[self.targ_traj_list_id_for_obj_0]

            # change colour of new target goal
            if not self.if_goal_visiualization:
                self._pb.changeVisualShape(self.goal_id_for_obj_0, -1, rgbaColor=[0, 0, 1, 0])
            else:
                self._pb.changeVisualShape(self.goal_id_for_obj_0, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached
            prev_goal_traj_list_id = (
                self.targ_traj_list_id_for_obj_0 - 1 if self.targ_traj_list_id_for_obj_0 > 0 else None
            )
            if prev_goal_traj_list_id is not None:
                if not self.if_goal_visiualization:
                    self._pb.changeVisualShape(
                        self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0]
                    )
                else:
                    self._pb.changeVisualShape(
                        self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5]
                    )

            return True

    def encode_TCP_frame_actions(self, actions, robot):
        encoded_actions = np.zeros(6)
        if robot.robot_lv == 'main_robot':
            tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_0) #this cur_tcp_orn_worldframe_robot——0 is got from get_step_data()
        elif robot.robot_lv == 'slave_robot':
            tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_1) #this cur_tcp_orn_worldframe_robot_1 is got from get_step_data()
            actions[0] = -actions[0]
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
            # be careful, since when movement_mode = "TyRz", the input only contains two elements, whatever
            # it is "TyRz" or "TxRz", the first element is always the Ty or Tx respectively.
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = 0.8 * self.max_action
            # par_scale = 0 * self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            # encoded_actions[2] = - self.max_action * 0.5
            # the actions[1] is the rotation z dim.
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TxRz":
            # translate the direction
            perp_scale = 0
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = 1.0 * self.max_action + actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            # the actions[1] is the rotation z dim.
            encoded_actions[5] += actions[1]
        elif self.movement_mode == "TxTyRz":

            # translate the direction
            # be careful, since when movement_mode = "TyRz", the input only contains two elements, whatever
            # it is "TyRz" or "TxRz", the first element is always the Ty or Tx respectively.
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = 1.0 * self.max_action - actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            # the actions[1] is the rotation z dim.
            # encoded_actions[2] = - self.max_action * 0.5
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
            if robot.if_main_robot == True:
                encoded_actions[1] = actions 
            elif robot.if_main_robot == False:
                encoded_actions[1] = -actions 
            encoded_actions[2] =  - self.max_action * 1
        elif self.movement_mode == "yz":
            if robot.if_main_robot == True:
                encoded_actions[1] = actions[0] 
            elif robot.if_main_robot == False:
                encoded_actions[1] = -actions[0] 
            encoded_actions[2] = actions[1] - self.max_action * 1
        elif self.movement_mode == "xz":
            if robot.if_main_robot == True:
                encoded_actions[0] = actions[0] 
            elif robot.if_main_robot == False:
                encoded_actions[0] = -actions[0] 
            encoded_actions[2] = actions[1] - self.max_action * 0.5
        elif self.movement_mode == "xyz":
            if robot.if_main_robot == True:
                encoded_actions[0] = self.max_action * 0.5
            elif robot.if_main_robot == False:
                encoded_actions[0] = -self.max_action * 0.5
            if robot.if_main_robot == True:
                encoded_actions[1] = actions[0] - self.max_action * 0.5
            elif robot.if_main_robot == False:
                encoded_actions[1] = actions[0] + self.max_action * 0.5
                
            encoded_actions[2] = actions[1] - self.max_action * 0.5
        elif self.movement_mode == "yRz":
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
        if self.movement_mode in ["y", 'xz', "yz", "xyz","yRz", "xyRz"]:
            encoded_actions = self.encode_work_frame_actions(actions, robot)
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
    
    def get_two_objs_current_states(self):
        (
            self.cur_obj_pos_worldframe_0, # used for calc xyz_obj_dist_to_goal for reward
            self.cur_obj_rpy_worldframe_0,
            self.cur_obj_orn_worldframe_0, # used for calc cos_tcp_dist_to_obj for reward
        ) = self.get_obj_id_pos_rpy_orn_worldframe(self.obj_id_0)

        (
            self.cur_obj_pos_worldframe_1, # used for calc xyz_obj_dist_to_goal for reward
            self.cur_obj_rpy_worldframe_1,
            self.cur_obj_orn_worldframe_1, # used for calc cos_tcp_dist_to_obj for reward
        ) = self.get_obj_id_pos_rpy_orn_worldframe(self.obj_id_1)

        (
            self.cur_obj_pos_workframe_0, # used for calc xyz_obj_dist_to_goal for reward
            self.cur_obj_rpy_workframe_0,
            self.cur_obj_orn_workframe_0, # used for calc cos_tcp_dist_to_obj for reward
        ) = self.get_obj_id_pos_rpy_orn_workframe(self.obj_id_0)

        (
            self.cur_obj_pos_workframe_1, # used for calc xyz_obj_dist_to_goal for reward
            self.cur_obj_rpy_workframe_1,
            self.cur_obj_orn_workframe_1, # used for calc cos_tcp_dist_to_obj for reward
        ) = self.get_obj_id_pos_rpy_orn_workframe(self.obj_id_1)

        (
            self.cur_obj_lin_vel_workframe_0,
            self.cur_obj_ang_vel_workframe_0,
        ) = self.get_obj_id_vel_workframe(self.obj_id_0)

        (
            self.cur_obj_lin_vel_workframe_1,
            self.cur_obj_ang_vel_workframe_1,
        ) = self.get_obj_id_vel_workframe(self.obj_id_1)

    def get_step_data(self):
        # For computing the reward as well as for  the observation.

        # get state of tcp and obj per step

        self.get_two_robots_current_states()
        self.get_two_objs_current_states()
        reward = self.get_reward()
        # get rl info
        done = self.termination()
        if self.if_apply_force_perturbation and self.xy_obj_0_dist_to_obj_1()>=self.stop_apply_force_pert_dist:
            self.apply_perturbation_for_objs()

        return reward, done

    def apply_perturbation_for_objs(self):
        if self.np_random.rand() < self.apply_pert_prob:
            force_mag=np.random.uniform(low = 1, high = 5)
            self.apply_random_force_base_for_obj_0(force_mag=force_mag)
        if self.np_random.rand() < self.apply_pert_prob:
            force_mag=np.random.uniform(low = 1, high = 5)
            self.apply_random_force_base_for_obj_1(force_mag=force_mag)

    def xyz_obj_scp_dist_to_tcp(self):
        """
        Cartesian distance from current side center points of the obejct to the current
        TCP point
        """
        # get the current obj_0 orientation matrix for robot_1
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)

        # get SCP vector of the object for robot_1
        # obj_scf_init_vector_objframe = np.array([-self.obj_length/2 + self.embed_dist, 0, 0])
        obj_scf_init_vector_objframe = np.array([-self.obj_length/2 + self.embed_dist/2, 0, 0])
        obj_scf_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scf_init_vector_objframe)
        obj_scf_cur_vector_worldframe = self.cur_obj_pos_worldframe_0 + obj_scf_cur_vector_objfrmae
        #  compute the SCP distance to the robot_1 tactip
        dist_1 = self.two_points_xy_dist(self.cur_tcp_pos_worldframe_robot_0, obj_scf_cur_vector_worldframe)

        # get the current obj_0 orientation matrix for robot_1
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        # get SCP vector of the object for robot_2
        # obj_scp_init_vector_objframe = np.array([self.obj_length/2 - self.embed_dist, 0, 0])
        obj_scp_init_vector_objframe = np.array([self.obj_length/2 - self.embed_dist/2, 0, 0])
        obj_scp_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scp_init_vector_objframe)
        obj_scp_cur_vector_worldframe = self.cur_obj_pos_worldframe_1 + obj_scp_cur_vector_objfrmae
        #  compute the SCP distance to the robot_2 tactip
        dist_2 = self.two_points_xy_dist(self.cur_tcp_pos_worldframe_robot_1, obj_scp_cur_vector_worldframe)
        
        return dist_1 + dist_2, dist_1, dist_2
    
    def cos_objs_dist_to_connection(self):
        vec_obj_0_to_obj_1 = self.cur_obj_pos_worldframe_1 - self.cur_obj_pos_worldframe_0
        
        obj_inner_scp_init_vector_0 = np.array([self.obj_length/2, 0, 0])
        obj_rot_matrix_0 = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        obj_rot_matrix_0 = np.array(obj_rot_matrix_0).reshape(3, 3)
        obj_inner_scp_cur_vector_objframe_0 = obj_rot_matrix_0.dot(obj_inner_scp_init_vector_0)
        # obj_inner_scp_cur_vector_worldframe_0 = self.cur_obj_pos_worldframe_0 + obj_inner_scp_cur_vector_objframe_0
        # c = [1,1,0]
        # self._pb.addUserDebugLine( obj_inner_scp_cur_vector_worldframe_0 ,  self.cur_obj_pos_worldframe_0, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        cos_sim_obj_0_to_connection = np.dot(vec_obj_0_to_obj_1, obj_inner_scp_cur_vector_objframe_0) / (
            np.linalg.norm(vec_obj_0_to_obj_1) * np.linalg.norm(obj_inner_scp_cur_vector_objframe_0)
        )
        cos_dist_obj_0_to_connection  = 1 - cos_sim_obj_0_to_connection
        obj_inner_scp_init_vector_1 = np.array([-self.obj_length/2, 0, 0])
        obj_rot_matrix_1 = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        obj_rot_matrix_1 = np.array(obj_rot_matrix_1).reshape(3, 3)
        obj_inner_scp_cur_vector_objframe_1 = obj_rot_matrix_1.dot(obj_inner_scp_init_vector_1)
        cos_sim_obj_1_to_connection = np.dot(-vec_obj_0_to_obj_1, obj_inner_scp_cur_vector_objframe_1) / (
            np.linalg.norm(-vec_obj_0_to_obj_1) * np.linalg.norm(obj_inner_scp_cur_vector_objframe_1)
        )
        cos_dist_obj_1_to_connection  = 1 - cos_sim_obj_1_to_connection
        total_cos_dist = cos_dist_obj_0_to_connection + cos_dist_obj_1_to_connection
        return total_cos_dist

    def cos_tcp_dist_to_obj(self):
        """
        Cos distance from current orientation of each TCP to the current
        orientation of each side (face to each TCP) of the object
        """
        # get current obj_0 for robot 1 orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)

        # get normal vector of object for robot 1
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
        # get current obj_1 for robot 2 orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)

        # get normal vector of object in the other direction for robot_2
        obj_init_vector_2 = np.array([-1, 0, 0])
        obj_vector_2 = obj_rot_matrix.dot(obj_init_vector_2)

        # get vector of tactip tip, directed through tip body
        tip_rot_matrix_robot_2 = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_1)
        tip_rot_matrix_robot_2 = np.array(tip_rot_matrix_robot_2).reshape(3, 3)
        tip_init_vector_robot_2 = np.array([1, 0, 0])
        tip_vector_robot_2 = tip_rot_matrix_robot_2.dot(tip_init_vector_robot_2)

        # get the cosine similarity/distance between the two vectors
        cos_sim_robot_2 = np.dot(obj_vector_2, tip_vector_robot_2) / (
            np.linalg.norm(obj_vector_2) * np.linalg.norm(tip_vector_robot_2)
        )
        cos_dist_robot_2 = 1 - cos_sim_robot_2
        cos_dist_robots = (cos_dist_robot + cos_dist_robot_2)/2

        
        return cos_dist_robots

    def termination(self):
        """
        Criteria for terminating an episode.
        """

        # check if near goal, change the goal if so
        objs_connection_pos_dist = self.xy_obj_0_dist_to_obj_1()
        xy_goal_dist_to_obj_0 = self.xy_goal_dist_to_obj_0()
        xy_goal_dist_to_obj_1 = self.xy_goal_dist_to_obj_1()
        self.obj_connection_goal_update_counter += 1
        if self.obj_connection_goal_update_counter == 10:
            if self.if_goal:
                self.make_goal()
            else:
                self.goal_pos_workframe_for_obj_0 *=0
                self.goal_pos_worldframe_for_obj_0 = self._workframe[:3]
                self.goal_pos_workframe_for_obj_1 *=0 
                self.goal_pos_worldframe_for_obj_1 = self._workframe[:3]
                
        if self.if_goal:
            if xy_goal_dist_to_obj_0 < self.goal_update_pos_dist:
                # update the goal (if not at end of traj)
                self.update_goal_for_obj_0()
            if xy_goal_dist_to_obj_1 < self.goal_update_pos_dist:
                # update the goal (if not at end of traj)
                self.update_goal_for_obj_1()

        if objs_connection_pos_dist < self.termination_pos_dist:
            return True
        if self._env_step_counter >= self._max_steps:
            return True
        return False
    


    def xy_obj_0_dist_to_obj_1(self):
        """
        xy L2 distance from the current obj position to another one's.
        """
        dist = np.linalg.norm(
            self.cur_obj_pos_worldframe_0[:2] - self.cur_obj_pos_worldframe_1[:2]
        )
        return dist

    def xy_goal_dist_to_obj_0(self):
        """
        xy L2 distance from the current obj_0 position to the goal.
        """
        dist = np.linalg.norm(
            self.cur_obj_pos_worldframe_0[:2] - self.goal_pos_worldframe_for_obj_0[:2]
        )

        return dist
        
    def xy_goal_dist_to_obj_1(self):
        """
        xy L2 distance from the current obj_1 position to the goal.
        """
        dist = np.linalg.norm(
            self.cur_obj_pos_worldframe_1[:2] - self.goal_pos_worldframe_for_obj_1[:2]
        )
        return dist


    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        xy_goal_dist_to_obj_0 = self.xy_goal_dist_to_obj_0()
        xy_goal_dist_to_obj_1 = self.xy_goal_dist_to_obj_1()
        obj_to_obj_pos_dist_xy = self.xy_obj_0_dist_to_obj_1()
        tip_obj_orn_dist = self.cos_tcp_dist_to_obj()
        objs_connection_orn_dist = self.cos_objs_dist_to_connection()
        # testing = self.draw_obj_side_center_points()
        # self.draw_obj_connection()
        _, TCPs_SCPs_pos_dist_1, TCPs_SCPs_pos_dist_2 = self.xyz_obj_scp_dist_to_tcp()

        if self.if_goal:
            W_obj_goal_pos = 1.0
            W_obj_to_obj_pos_dist_xy = 0
        else:
            W_obj_goal_pos = 0
            W_obj_to_obj_pos_dist_xy = 1.0
        W_tip_obj_orn = 1.0
        W_objs_connection_orn = 2.0
        w_TCPs_SCPs_pos_1 = 1.0
        w_TCPs_SCPs_pos_2 = 1.0

        reward = -(
            (W_obj_goal_pos * xy_goal_dist_to_obj_0)
            + (W_obj_goal_pos * xy_goal_dist_to_obj_1)
            + (W_obj_to_obj_pos_dist_xy * obj_to_obj_pos_dist_xy)
            + (W_objs_connection_orn * objs_connection_orn_dist)
            + (W_tip_obj_orn * tip_obj_orn_dist)
            + (w_TCPs_SCPs_pos_1 * TCPs_SCPs_pos_dist_1)
            + (w_TCPs_SCPs_pos_2 * TCPs_SCPs_pos_dist_2)
        )
        return reward

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

                *self.cur_obj_pos_workframe_0,
                *self.cur_obj_rpy_workframe_0,
                *self.cur_obj_pos_workframe_1,
                *self.cur_obj_rpy_workframe_1,

                *self.cur_obj_lin_vel_workframe_0,
                *self.cur_obj_ang_vel_workframe_0,
                *self.cur_obj_lin_vel_workframe_1,
                *self.cur_obj_ang_vel_workframe_1,

            ]
        )


        return observation

    def get_extended_feature_array(self):

        obj_xy_pos_workframe_0 = self.cur_obj_pos_workframe_0[:2]
        obj_Rz_workframe_0 = self.cur_obj_rpy_workframe_0[2]
        obj_xy_pos_workframe_0 *= self.if_use_obj_xy_info
        obj_Rz_workframe_0 *= self.if_use_obj_Rz_info
        obj_xy_pos_workframe_1 = self.cur_obj_pos_workframe_1[:2]
        obj_Rz_workframe_1 = self.cur_obj_rpy_workframe_1[2]
        obj_xy_pos_workframe_1 *= self.if_use_obj_xy_info
        obj_Rz_workframe_1 *= self.if_use_obj_Rz_info
        tcp_xy_pos_workframe_robot = self.cur_tcp_pos_workframe_robot_0[:2]
        tcp_Rz_workframe_robot = self.cur_tcp_rpy_workframe_robot_0[2]
        tcp_xy_pos_workframe_robot_1 = self.cur_tcp_pos_workframe_robot_1[:2]
        tcp_Rz_workframe_robot_1 = self.cur_tcp_rpy_workframe_robot_1[2]
        goal_xy_workframe_for_obj_0 = self.goal_pos_workframe_for_obj_0[:2]
        goal_xy_workframe_for_obj_1 = self.goal_pos_workframe_for_obj_1[:2]

        feature_array = np.array(
            [
                *tcp_xy_pos_workframe_robot,
                tcp_Rz_workframe_robot,
                *tcp_xy_pos_workframe_robot_1,
                tcp_Rz_workframe_robot_1,
                *obj_xy_pos_workframe_0,  # should be 0 at testing
                obj_Rz_workframe_0,    # should be 0 at testing
                *obj_xy_pos_workframe_1,  # should be 0 at testing
                obj_Rz_workframe_1,    # should be 0 at testing
                *goal_xy_workframe_for_obj_0,
                *goal_xy_workframe_for_obj_1,
            ]
        )
        # print("tcp_xy_pos_workframe_robot:",tcp_xy_pos_workframe_robot)
        # print("tcp_Rz_workframe_robot:",tcp_Rz_workframe_robot)
        # print("tcp_xy_pos_workframe_robot_2:",tcp_xy_pos_workframe_robot_1)
        # print("tcp_Rz_workframe_robot_2:",tcp_Rz_workframe_robot_1)
        # print("obj_xy_pos_workframe:",obj_xy_pos_workframe_0)
        # print("obj_Rz_workframe:",obj_Rz_workframe_0)
        # print("obj_xy_pos_workframe_1:",obj_xy_pos_workframe_1)
        # print("obj_Rz_workframe_1:",obj_Rz_workframe_1)
        # print("goal_xy_workframe_for_obj_0:",goal_xy_workframe_for_obj_0)
        # print("goal_xy_workframe_for_obj_1:",goal_xy_workframe_for_obj_1)
        # print("env_step:",self._env_step_counter)
        return feature_array

    def apply_random_force_base_for_obj_0(self, force_mag=0.1):
        """
        Apply a random force to the object perpendicular to the object long side.
        """
        # calculate force, the position is the inner middle side of the object
        force_pos_0 = self.cur_obj_pos_worldframe_0 + np.array(
            [
                -self.obj_length / 2,
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_length / 2,
                0,
            ]
        )
        cur_obj_rot_mat_worldframe_0 = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        cur_obj_rot_mat_worldframe_0 = np.array(cur_obj_rot_mat_worldframe_0).reshape(3, 3)
        if self.per_direcction_for_obj_0 is None:
            obj_perp_init_vector = self.np_random.choice([-1, 1]) * np.array([0, 1, 0])
        else:
            obj_perp_init_vector = self.per_direcction_for_obj_0 * np.array([0, 1, 0])
        force_dir = cur_obj_rot_mat_worldframe_0.dot(obj_perp_init_vector)
        force = force_dir * force_mag
        # apply force
        self._pb.applyExternalForce(
            self.obj_id_0, -1, force, force_pos_0, flags=self._pb.WORLD_FRAME
        )
        # plot force
        draw_vector(force_pos_0, force_dir)

    def apply_random_force_base_for_obj_1(self, force_mag=0.1):
        """
        Apply a random force to the object perpendicular to the object long side.
        """

        # calculate force
        force_pos_1 = self.cur_obj_pos_worldframe_1 + np.array(
            [
                self.obj_length / 2,
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_length / 2,
                0,
            ]
        )
        cur_obj_rot_mat_worldframe_1 = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        cur_obj_rot_mat_worldframe_1 = np.array(cur_obj_rot_mat_worldframe_1).reshape(3, 3)

        if self.per_direcction_for_obj_1 is None:
            obj_perp_init_vector = self.np_random.choice([-1, 1]) * np.array([0, 1, 0])
        else:
            obj_perp_init_vector = self.per_direcction_for_obj_1 * np.array([0, 1, 0])

        force_dir = cur_obj_rot_mat_worldframe_1.dot(obj_perp_init_vector)
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalForce(
            self.obj_id_1, -1, force, force_pos_1, flags=self._pb.WORLD_FRAME
        )

        # plot force
        draw_vector(force_pos_1, force_dir)

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


    def draw_obj_connection(self):
        obj_inner_scp_init_vector_0 = np.array([-self.obj_length/2, 0, 0])
        obj_inner_scp_init_vector_1 = np.array([self.obj_length/2, 0, 0])
        c = [0,1,0]
        self._pb.addUserDebugLine( self.cur_obj_pos_worldframe_1 ,  self.cur_obj_pos_worldframe_0, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        obj_inner_scp_cur_vector_objfrmae_0 = obj_rot_matrix.dot(obj_inner_scp_init_vector_0)
        obj_inner_scp_cur_vector_worldframe_0 = self.cur_obj_pos_worldframe_0 + obj_inner_scp_cur_vector_objfrmae_0
        c = [1,1,0]
        self._pb.addUserDebugLine( obj_inner_scp_cur_vector_worldframe_0 ,  self.cur_obj_pos_worldframe_0, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        obj_inner_scp_cur_vector_objframe_1 = obj_rot_matrix.dot(obj_inner_scp_init_vector_1)
        obj_inner_scp_cur_vector_worldframe_1 = self.cur_obj_pos_worldframe_1 + obj_inner_scp_cur_vector_objframe_1
        c = [1,1,0]
        self._pb.addUserDebugLine( obj_inner_scp_cur_vector_worldframe_1 ,  self.cur_obj_pos_worldframe_1, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

    def draw_obj_side_center_points(self):
        '''
        Test the obj_scp_cur_vector_worldframe and obj_scf_cur_vector_worldframe
        '''
        # get the current obj_1 for robot_2 orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_1)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        # get SCP vector of the object for robot_2
        obj_scp_init_vector_objframe = np.array([self.obj_length/2, 0, 0])
        obj_scp_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scp_init_vector_objframe)
        obj_scp_cur_vector_worldframe = self.cur_obj_pos_worldframe_1 + obj_scp_cur_vector_objfrmae

        obj_scq_init_vector_objframe = np.array([self.obj_length/2, self.obj_length/2, 0])
        obj_scm_init_vector_objframe = np.array([self.obj_length/2, 0 ,  self.obj_length/2])
        obj_scn_init_vector_objframe = np.array([2 * self.obj_length/2, 0 , 0])
        assist_point_list_2 = []
        for point in [obj_scq_init_vector_objframe, obj_scm_init_vector_objframe, obj_scn_init_vector_objframe]:
            obj_cur_vector_objfrmae = obj_rot_matrix.dot(point)
            obj_cur_vector_worldframe = self.cur_obj_pos_worldframe_1 + obj_cur_vector_objfrmae
            assist_point_list_2.append(obj_cur_vector_worldframe)
        # draw for debugging
        # make the object transparent
        self._pb.changeVisualShape(
          self.obj_id_1  , -1, rgbaColor=[1, 1, 1, 0.3]
        )
        colors_1 = [[1,0,0], [0,1,0], [0,0,1]]
        for point,c in zip(assist_point_list_2,colors_1):
            self._pb.addUserDebugLine(obj_scp_cur_vector_worldframe, point, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        # get the current obj_0 for robot_1 orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe_0)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        # get SCP f vector of the object for robot_1
        obj_scf_init_vector_objframe = np.array([-self.obj_length/2, 0, 0])
        obj_scf_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scf_init_vector_objframe)
        obj_scf_cur_vector_worldframe = self.cur_obj_pos_worldframe_0 + obj_scf_cur_vector_objfrmae

        obj_scg_init_vector_objframe = np.array([-self.obj_length/2, -self.obj_length/2, 0])
        obj_sch_init_vector_objframe = np.array([-self.obj_length/2, 0 ,  -self.obj_length/2])
        obj_sce_init_vector_objframe = np.array([-2 * self.obj_length/2, 0 , 0])
        assist_point_list_1 = []
        for point in [obj_scg_init_vector_objframe, obj_sch_init_vector_objframe, obj_sce_init_vector_objframe]:
            obj_cur_vector_objfrmae = obj_rot_matrix.dot(point)
            obj_cur_vector_worldframe = self.cur_obj_pos_worldframe_0 + obj_cur_vector_objfrmae
            assist_point_list_1.append(obj_cur_vector_worldframe)
        colors_1 = [[1,.5,0], [.5,0,1], [0,1,1]]

        self._pb.changeVisualShape(
          self.obj_id_0  , -1, rgbaColor=[1, 1, 1, 0.3]
        )

        for point,c in zip(assist_point_list_1,colors_1):
            self._pb.addUserDebugLine(obj_scf_cur_vector_worldframe, point, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)


    def draw_obj_workframe(self):
        self._pb.changeVisualShape(
          self.obj_id  , -1, rgbaColor=[1, 1, 1, 0.3]
        )
        return super().draw_obj_workframe()
