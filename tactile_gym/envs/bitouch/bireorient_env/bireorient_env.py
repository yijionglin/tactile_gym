import os, sys
import numpy as np
from tactile_gym.assets import add_assets_path
from tactile_gym.envs.bitouch.bireorient_env.poses import (
    rest_poses_dict,
    EEs_poses_sets,
)
from tactile_gym.envs.bitouch.base_bitouch_object_env import BaseBitouchObjectEnv
import cv2
from ipdb import set_trace

class BireorientEnv(BaseBitouchObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},

    ):
        # env specific values
        self.env_name = "bireorient"
        self.termination_pos_dist = 0.025
        self.termination_orn_dist = 0.040
        self.visualise_goal = False
        self.goal_line_id = None
        self.embed_dist = 0.005
        self.obj_width = 0.06
        self.obj_length = 0.06
        self.obj_height = 0.1
        self.eval_ang_dist_list = []
        self.if_draw_indicator_line = True
        self.if_draw_indicator_lines = False
        self.if_use_obj_xy_info = 0
        self.if_use_obj_Rz_info = 0

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.0, 0.0, 0.1 + self.obj_height/2, -np.pi, 0.0, np.pi/2])
        TCP_lims_robot_0 = np.zeros(shape=(6, 2))
        TCP_lims_robot_0[0, 0], TCP_lims_robot_0[0, 1] = -0.15, 0.05  # x lims
        TCP_lims_robot_0[1, 0], TCP_lims_robot_0[1, 1] = -0.18, 0.18  # y lims
        TCP_lims_robot_0[2, 0], TCP_lims_robot_0[2, 1] = -0.12, 0.12  # z lims
        TCP_lims_robot_0[3, 0], TCP_lims_robot_0[3, 1] = -0.0, 0.0  # roll lims
        TCP_lims_robot_0[4, 0], TCP_lims_robot_0[4, 1] = -0.0, 0.0  # pitch lims
        TCP_lims_robot_0[5, 0], TCP_lims_robot_0[5, 1] = -100 * np.pi / 180, 100 * np.pi / 180  # yaw lims
        TCP_lims_robot_1 = TCP_lims_robot_0.copy()
        TCP_lims_robot_1[0, 0], TCP_lims_robot_1[0, 1] = 0.01, 0.10  # x lims
        TCP_lims_robot_1[5, 0], TCP_lims_robot_1[5, 1] = (-100-180) * np.pi / 180, (100-180) * np.pi / 180  # yaw lims
        TCP_lims_list = [TCP_lims_robot_0, TCP_lims_robot_1]
        env_params["tcp_lims"] = TCP_lims_list
        self.rand_obj_mass = env_params["rand_obj_mass"]
        self.traj_type = env_params["traj_type"]

        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = True
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]][tactile_sensor_params["type"]]
        robot_arm_params["base_pos_and_init_pos"] = EEs_poses_sets[robot_arm_params["type"]]
        robot_arm_params["base_rpy_and_init_rpy"] = EEs_poses_sets[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"
        
        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "fixed"
        tactile_sensor_params["dynamics"] = {"stiffness": 500, "damping": 0, "friction": 10.0}
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
        super(BireorientEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def load_goal_line(self):
        self.goal_line_id = self._pb.loadURDF(
            self.goal_line_path, self.init_obj_pos, self.init_obj_orn, useFixedBase=True
        )
        self._pb.setCollisionFilterGroupMask(self.goal_line_id, -1, 0, 0)

    def apply_constraints(self):
        
        child_pos_1 = [-self.obj_length/2 + self.embed_dist, 0 , 0 ]
        obj_to_const_id = self.obj_id
        self.obj_tip_constraint_id = self._pb.createConstraint(
            self.embodiment_0.robot_id,
            self.embodiment_0.arm.TCP_link_id,
            obj_to_const_id,
            -1,
            self._pb.JOINT_POINT2POINT,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=child_pos_1,
            parentFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
        )
        child_pos_2 = [self.obj_length/2 - self.embed_dist, 0 , 0 ]
        self.obj_tip_constraint_id_2 = self._pb.createConstraint(
            self.embodiment_1.robot_id,
            self.embodiment_1.arm.TCP_link_id,
            obj_to_const_id,
            -1,
            self._pb.JOINT_POINT2POINT,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=child_pos_2,
            parentFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
        )
        

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
        self.init_obj_pos =self._workframe[:3]
        # The object initial orientation needs to be consistent with the work frame
        self.init_obj_rpy = self._workframe[3:]
        self.init_obj_orn = self._pb.getQuaternionFromEuler(self.init_obj_rpy)
        # get paths
        self.object_path = add_assets_path("bitouch/bireorient_obj/cube_6.urdf")
        self.goal_line_path = add_assets_path("bitouch/bireorient_obj/goal_indicator.urdf")
        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")


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
        # self.apply_constraints()
        self.reset_counter = 0

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
        self.reset_task() #  used in marl lift task, not in push task
        self.update_workframe() # not used in pushing task
        # make room for the object
        reset_tcp_pose_0_workframe = np.array([x for x in [-0.12,0,0]] + [x for x in  self.embodiment_0.update_init_rpy]) 
        reset_tcp_pose_1_workframe = np.array([x for x in [0.12,0,0]] + [x for x in  self.embodiment_1.update_init_rpy])
        reset_tcp_pose_0_worldframe = self.workframe_to_worldframe(reset_tcp_pose_0_workframe)
        reset_tcp_pose_1_worldframe = self.workframe_to_worldframe(reset_tcp_pose_1_workframe)
        self.embodiment_0.reset(reset_tcp_pose=reset_tcp_pose_0_worldframe)
        # self.embodiment_0.arm.draw_TCP()
        self.embodiment_1.reset(reset_tcp_pose=reset_tcp_pose_1_worldframe)
        self.reset_object()
        reset_tcp_pose_0_worldframe = self.workframe_to_worldframe(self.embodiment_0.update_init_pose)
        reset_tcp_pose_1_worldframe = self.workframe_to_worldframe(self.embodiment_1.update_init_pose)
        self.embodiment_0.reset(reset_tcp_pose=reset_tcp_pose_0_worldframe)
        self.embodiment_1.reset(reset_tcp_pose=reset_tcp_pose_1_worldframe)
        # define a new goal position based on init pose of object
        self.make_goal()
        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()
        # get the starting observation
        self._observation = self.get_observation()
        
        return self._observation

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        self.eval_ang_dist_list = []
        self._pb.removeBody(self.obj_id)
        if self.goal_line_id is not None:
            self._pb.removeBody(self.goal_line_id)
        try:
            cube_urdf_id = self.np_random.randint(5,15)
        except AttributeError as e:
            print(f"AttributeError: {e}")
            print("Using self.np_random.integers instead.")
            cube_urdf_id = self.np_random.integers(5,15)
        object_path = "bitouch/bireorient_obj/cube_" + str(cube_urdf_id) + ".urdf"
        self.object_path = add_assets_path(object_path)
        self.obj_length = 0.01 * cube_urdf_id
        self.fake_goal_x_workframe  = self.np_random.choice([0, 1]) * self.obj_length/2
        obj_init_y_offset = 0
        self.init_obj_pos[1] =  obj_init_y_offset
        self.obj_id = self._pb.loadURDF(
            self.object_path, self.init_obj_pos, self.init_obj_orn
        )
        self._pb.changeVisualShape(self.obj_id, -1, rgbaColor=[1-(cube_urdf_id-5)/10, 0,(cube_urdf_id-5)/10 , 1])
        self.embodiment_0.update_init_pose[0] = -self.obj_length/2 
        self.embodiment_1.update_init_pose[0] = self.obj_length/2 
        self.goal_line_id = self._pb.loadURDF(
            self.goal_line_path, self.init_obj_pos, self.init_obj_orn,useFixedBase=True
        )
        self._pb.setCollisionFilterGroupMask(self.goal_line_id, -1, 0, 0)
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
        obj_mass = 0.05
        self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)
        if self.rand_obj_mass:
            obj_mass = self.np_random.uniform(0.4, 0.8)
            self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)

    def load_trajectory(self):

        # relatively easy traj
        if self.traj_type == "rotation":
            self.traj_n_points = 5
        else:
            self.traj_n_points = 10
        self.traj_spacing = 0.025
        self.traj_ids = []
        # if self.traj_type == 'rotation_only':
        for i in range(int(self.traj_n_points)):
            # just loading goals here, no placement yet
            
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                [pos[0],0,pos[1]],
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

        if self.traj_type == "rotation":
            self.update_trajectory_rotation()
        elif self.traj_type == "rotation_only":
            if self.if_draw_indicator_line :
                self._pb.removeAllUserDebugItems()
            self.update_trajectory_rotation_only()
        else:
            sys.exit("Incorrect traj_type specified: {}".format(self.traj_type))
        # calc orientation to place object at
        if self.traj_type not in ["rotation", 'rotation_only']:
            self.traj_rpy_workframe[:, 2] = np.gradient(
                self.traj_pos_workframe[:, 1], self.traj_spacing
            )
        else:
            angle_spacing = self.largest_angle / self.traj_n_points
            for i in range(int(self.traj_n_points)):
                self.traj_rpy_workframe[i, 2] = i*angle_spacing


        self.indicator_lines_id_list = []
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
            # self.draw_goal_frame(self.traj_ids[i])
            if self.traj_type == "rotation_only" and self.if_draw_indicator_lines == True:
                line_id = self.draw_indicator_line(self.traj_ids[i])
                self.indicator_lines_id_list.append(line_id)
        if self.if_draw_indicator_line:
            self.draw_goal_frame(self.obj_id)
            if self.traj_type == "rotation_only":
                self._pb.resetBasePositionAndOrientation(
                    self.goal_line_id, pos_worldframe, orn_worldframe
                )
    def update_trajectory_rotation(self):
        """
        Generates smooth trajectory of goals. In the lifting, the z is the input and the y is the output
        of the traj mapping.
        """
        # NOTE: these pose are calc in work frame, not world frame or tcp frame.
        # randomly pick traj direction
        traj_ang = self.np_random.uniform(0 ,0)
        self.largest_angle = self.np_random.uniform(-np.pi/4 ,np.pi/4)
        init_offset_x =  self.obj_length / 2
        init_offset_z = self.obj_height/2
        for i in range(int(self.traj_n_points)):

            dir_z = np.cos(traj_ang)
            dir_y = np.sin(traj_ang)
            dist = i * self.traj_spacing

            x = init_offset_x 
            y = dist * dir_y
            z = - (init_offset_z + dist * dir_z)
            self.traj_pos_workframe[i] = [x, y, z]

    def update_trajectory_rotation_only(self):
        """
        Generates smooth trajectory of goals. In the lifting, the z is the input and the y is the output
        of the traj mapping.
        """
        # NOTE: these pose are calc in work frame, not world frame or tcp frame.
        # randomly pick traj direction
        traj_ang = 0
        direction =self.np_random.choice([-1,1])
        self.largest_angle = direction * self.np_random.uniform(-np.pi/2 ,-np.pi/6)
        if direction==-1:
            self.largest_angle = self.np_random.uniform(-np.pi/2 ,-np.pi/6)
        else:
            self.largest_angle = self.np_random.uniform(np.pi/2 ,np.pi/6)

        init_offset_x =  0
        init_offset_z = self.obj_height/2
        for i in range(int(self.traj_n_points)):

            dir_z = np.cos(traj_ang)
            dir_y = np.sin(traj_ang)
            dist = i * self.traj_spacing

            x = init_offset_x 
            y = dist * dir_y
            z = - (init_offset_z + 0.15)
            self.traj_pos_workframe[i] = [x, y, z]

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """

        self.goal_on_hold_step = 0
        self.embed_dist = self.np_random.uniform(0.003,0.008)
        # It turns out the agent can learn the task without obj pose info.
        # self.if_use_obj_xy_info = self.np_random.choice([0, 1])
        # self.if_use_obj_Rz_info = self.np_random.choice([0, 1])
        self.if_use_obj_xy_info = 0
        self.if_use_obj_Rz_info = 0
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
            self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id] # for getting the extra feature and for getting the oracle obs
            self.goal_orn_workframe = self.traj_orn_workframe[self.targ_traj_list_id] # not used
            self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id] # for getting the extra feature and for getting the oracle obs
            if self.traj_type != "rotation_only":
                self._pb.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 0, 1, 0.5])
                # change colour of goal just reached, which is in red.
                prev_goal_traj_list_id = (
                    self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None
                )
                if prev_goal_traj_list_id is not None:
                    self._pb.changeVisualShape(
                        self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5]
                    )
            else:
                if self.if_draw_indicator_lines:
                    self.indicator_line_id = (self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None)
                    if self.indicator_line_id is not None:
                        self._pb.removeUserDebugItem(self.indicator_line_id)
                
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
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)
            par_scale = 0
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TxRz":
            perp_scale = 0
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)
            par_scale = 1.0 * self.max_action + actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == "TxTyRz":
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)
            par_scale = actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)
            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[2]
            if self.largest_angle >0:
                motion_dir = -1  
                encoded_actions[1] *= motion_dir
                encoded_actions[5] *= motion_dir
        else:
            assert "Error: input movement mode has not been defined"
        return encoded_actions

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel. For MARL.
        """
        # get image from sensor
        tactile_obs_1 = self.embodiment_0.get_tactile_observation()
        tactile_obs_2 = self.embodiment_1.get_tactile_observation()
        if self.largest_angle>0:
            tactile_obs_1 = cv2.flip(tactile_obs_1, 0)
            tactile_obs_2 = cv2.flip(tactile_obs_2, 0)
        observation = np.concatenate([tactile_obs_1, tactile_obs_2], axis=1)
        observation = observation[..., np.newaxis]
        return observation

        
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

    def get_step_data(self):
        # For computing the reward as well as for  the observation.
        # get state of tcp and obj per step
        self.get_two_robots_current_states()
        self.get_obj_current_state()
        reward = self.get_reward()
        # get rl info
        done = self.termination()

        return reward, done
    
    def xyz_obj_scp_dist_to_tcp(self):
        """
        Cartesian distance from current side center points of the obejct to the current
        TCP point
        """
        # get the current obj orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        # get SCP vector of the object for robot_1
        obj_scp_init_vector_objframe = np.array([self.obj_length/2 - self.embed_dist/2, 0, 0])
        obj_scp_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scp_init_vector_objframe)
        obj_scp_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_scp_cur_vector_objfrmae
        #  compute the SCP distance to the robot_1 tactip
        dist_2 = self.two_points_xy_dist(self.cur_tcp_pos_worldframe_robot_1, obj_scp_cur_vector_worldframe)
        # get SCP vector of the object for robot_1
        obj_scf_init_vector_objframe = np.array([-self.obj_length/2 + self.embed_dist/2, 0, 0])
        obj_scf_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scf_init_vector_objframe)
        obj_scf_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_scf_cur_vector_objfrmae
        #  compute the SCP distance to the robot_1 tactip
        dist_1 = self.two_points_xy_dist(self.cur_tcp_pos_worldframe_robot_0, obj_scf_cur_vector_worldframe)
        return dist_1 + dist_2, dist_1, dist_2

    def draw_obj_workframe(self):
        self._pb.changeVisualShape(
          self.obj_id  , -1, rgbaColor=[1, 1, 1, 0.3]
        )
        return super().draw_obj_workframe()
    def draw_obj_side_center_points(self):
        '''
        Test the obj_scp_cur_vector_worldframe and obj_scf_cur_vector_worldframe
        '''
        # get the current obj orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)

        # get SCP vector of the object for robot_1
        obj_scp_init_vector_objframe = np.array([self.obj_length/2, 0, 0])
        obj_scp_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scp_init_vector_objframe)
        obj_scp_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_scp_cur_vector_objfrmae

        obj_scq_init_vector_objframe = np.array([self.obj_length/2, self.obj_length/2, 0])
        obj_scm_init_vector_objframe = np.array([self.obj_length/2, 0 ,  self.obj_length/2])
        obj_scn_init_vector_objframe = np.array([2 * self.obj_length/2, 0 , 0])
        assist_point_list_2 = []
        for point in [obj_scq_init_vector_objframe, obj_scm_init_vector_objframe, obj_scn_init_vector_objframe]:
            obj_cur_vector_objfrmae = obj_rot_matrix.dot(point)
            obj_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_cur_vector_objfrmae
            assist_point_list_2.append(obj_cur_vector_worldframe)
        # draw for debugging
        # make the object transparent
        self._pb.changeVisualShape(
          self.obj_id  , -1, rgbaColor=[1, 1, 1, 0.3]
        )
        colors_1 = [[1,0,0], [0,1,0], [0,0,1]]
        for point,c in zip(assist_point_list_2,colors_1):
            self._pb.addUserDebugLine(obj_scp_cur_vector_worldframe, point, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        # get SCP f vector of the object for robot_1
        obj_scf_init_vector_objframe = np.array([-self.obj_length/2, 0, 0])
        obj_scf_cur_vector_objfrmae = obj_rot_matrix.dot(obj_scf_init_vector_objframe)
        obj_scf_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_scf_cur_vector_objfrmae

        obj_scg_init_vector_objframe = np.array([-self.obj_length/2, -self.obj_length/2, 0])
        obj_sch_init_vector_objframe = np.array([-self.obj_length/2, 0 ,  -self.obj_length/2])
        obj_sce_init_vector_objframe = np.array([-2 * self.obj_length/2, 0 , 0])
        assist_point_list_1 = []
        for point in [obj_scg_init_vector_objframe, obj_sch_init_vector_objframe, obj_sce_init_vector_objframe]:
            obj_cur_vector_objfrmae = obj_rot_matrix.dot(point)
            obj_cur_vector_worldframe = self.cur_obj_pos_worldframe + obj_cur_vector_objfrmae
            assist_point_list_1.append(obj_cur_vector_worldframe)
        colors_1 = [[1,.5,0], [.5,0,1], [0,1,1]]
        for point,c in zip(assist_point_list_1,colors_1):
            self._pb.addUserDebugLine(obj_scf_cur_vector_worldframe, point, c, parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)


    def cos_tcp_dist_to_obj(self):
        """
        Cos distance from current orientation of each TCP to the current
        orientation of each side (face to each TCP) of the object
        """
        # get current obj orientation matrix
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_obj_orn_worldframe)
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
        # get normal vector of object in the other direction for robot_1
        obj_init_vector_2 = np.array([-1, 0, 0])
        obj_vector_2 = obj_rot_matrix.dot(obj_init_vector_2)
        # get vector of tactip tip, directed through tip body
        tip_rot_matrix_robot_1 = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe_robot_1)
        tip_rot_matrix_robot_1 = np.array(tip_rot_matrix_robot_1).reshape(3, 3)
        tip_init_vector_robot_1 = np.array([1, 0, 0])
        tip_vector_robot_1 = tip_rot_matrix_robot_1.dot(tip_init_vector_robot_1)
        # get the cosine similarity/distance between the two vectors
        cos_sim_robot_1 = np.dot(obj_vector_2, tip_vector_robot_1) / (
            np.linalg.norm(obj_vector_2) * np.linalg.norm(tip_vector_robot_1)
        )
        cos_dist_robot_1 = 1 - cos_sim_robot_1
        cos_dist_robots = (cos_dist_robot + cos_dist_robot_1)/2
        
        return cos_dist_robots

    def termination(self):
        """
        Criteria for terminating an episode.
        """
        # check if near goal, change the goal if so
        if self.traj_type != "rotation_only":
            obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
            if obj_goal_pos_dist < self.termination_pos_dist:
                goal_updated = self.update_goal()
                if not goal_updated:
                    return True
            if self._env_step_counter >= self._max_steps:
                return True
        else:
            obj_goal_orn_dist = self.orn_obj_dist_to_goal()
            
            if obj_goal_orn_dist < self.termination_orn_dist:

                goal_updated = self.update_goal()

                if not goal_updated:
                    self.goal_on_hold_step += 1
                    self.embed_dist = 0
                    cur_obj_rpy_worldframe = self._pb.getEulerFromQuaternion(self.cur_obj_orn_worldframe)
                    goal_obj_worldframe = self._pb.getEulerFromQuaternion(self.goal_orn_worldframe)
                    eval_ang_dist = goal_obj_worldframe[2] - cur_obj_rpy_worldframe[2]
                    self.eval_ang_dist_list.append(eval_ang_dist)
                    if self.goal_on_hold_step >= 5:
                        return True
            # terminate when max ep len reached
            if self._env_step_counter >= self._max_steps:
                return True

        return False

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        +1 is given for each goal reached.
        This is calculated before termination called as that will update the goal.
        """
        assert "Error: not defined in marl env."
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        obj_goal_pos_dist_z = self.xyz_obj_dist_to_goal()
        obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        tip_obj_orn_dist = self.cos_tcp_dist_to_obj()
        _, TCPs_SCPs_pos_dist_1, TCPs_SCPs_pos_dist_2 = self.xyz_obj_scp_dist_to_tcp()
        # weights for rewards
        W_obj_goal_pos_z = 2.0
        W_obj_goal_orn = 1.0
        W_tip_obj_orn = 2.0
        w_TCPs_SCPs_pos_1 = 3.0
        w_TCPs_SCPs_pos_2 = 3.0
        # sum rewards with multiplicative factors
        reward = -(
            (W_obj_goal_pos_z * obj_goal_pos_dist_z)
            + (W_obj_goal_orn * obj_goal_orn_dist)
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
                *self.cur_obj_pos_workframe,
                *self.cur_obj_rpy_workframe,
                *self.cur_obj_lin_vel_workframe,
                *self.cur_obj_ang_vel_workframe,
                *self.goal_pos_workframe,
                *self.goal_rpy_workframe,
            ]
        )

        return observation

    def get_extended_feature_array(self):
        # get sim info on TCP
        obj_pos_workframe, obj_rpy_workframe, _ = self.get_obj_pos_rpy_orn_workframe()
        obj_xy_pos_workframe = obj_pos_workframe[:2]
        obj_Rz_workframe = obj_rpy_workframe[2]
        obj_xy_pos_workframe *= self.if_use_obj_xy_info
        obj_Rz_workframe *= self.if_use_obj_Rz_info

        tcp_pos_workframe_robot_0 = self.cur_tcp_pos_workframe_robot_0
        tcp_rpy_workframe_robot_0 = self.cur_tcp_rpy_workframe_robot_0
        tcp_pos_workframe_robot_1 = self.cur_tcp_pos_workframe_robot_1
        tcp_rpy_workframe_robot_1 = self.cur_tcp_rpy_workframe_robot_1

        tcp_xy_pos_workframe_robot = tcp_pos_workframe_robot_0[:2]
        tcp_Rz_workframe_robot = tcp_rpy_workframe_robot_0[2]
        tcp_xy_pos_workframe_robot_1 = tcp_pos_workframe_robot_1[:2]
        tcp_Rz_workframe_robot_1 = tcp_rpy_workframe_robot_1[2]
        goal_x_workframe = self.fake_goal_x_workframe
        goal_Rz_workframe = self.goal_rpy_workframe[2]
        if self.largest_angle>0:
            tcp_xy_pos_workframe_robot[1] *= -1
            tcp_Rz_workframe_robot *= -1
            tcp_xy_pos_workframe_robot_1[1] *= -1
            tcp_Rz_workframe_robot_1 *= -1
            goal_Rz_workframe *= -1

        feature_array = np.array(
            [
                *tcp_xy_pos_workframe_robot,
                tcp_Rz_workframe_robot,
                *tcp_xy_pos_workframe_robot_1,
                tcp_Rz_workframe_robot_1,
                *obj_xy_pos_workframe,  # should be 0 at testing
                obj_Rz_workframe,    # should be 0 at testing
                goal_x_workframe,   # should be 0 at testing
                goal_Rz_workframe,
                self.embed_dist,    # should be defined at testing
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

    def draw_goal_frame(self, id):
        self._pb.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=id)
        self._pb.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=id)
        self._pb.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=id)
        # , lifeTime=1

    def draw_indicator_line(self, id):
        line_id = self._pb.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=id)
        return line_id