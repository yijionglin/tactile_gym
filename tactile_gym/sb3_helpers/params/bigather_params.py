from tactile_gym.sb3_helpers.params.default_params_bitouch import env_args
from tactile_gym.sb3_helpers.params.default_params_bitouch import rl_params_ppo
from tactile_gym.sb3_helpers.params.default_params_bitouch import ppo_params
from tactile_gym.sb3_helpers.params.default_params_bitouch import rl_params_sac
from tactile_gym.sb3_helpers.params.default_params_bitouch import sac_params


env_args["env_params"]["max_steps"] = 300
# env_args["env_params"]["observation_mode"] = "oracle"
env_args["env_params"]["observation_mode"] = "tactile_and_feature"
# env_args["env_params"]["observation_mode"] = "visual_and_feature"
# env_args["env_params"]["observation_mode"] = "visuotactile_and_feature"
env_args["env_params"]["rand_obj_mass"] = False
env_args["env_params"]["rand_init_orn"] = False
env_args["env_params"]["traj_type"] = "obj_straight_connection"




env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["Ty", "Rz", "Ty", "Rz"]

rl_params_ppo["env_id"] = "bigather-v0"
rl_params_ppo["total_timesteps"] = int(1e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "bigather-v0"
rl_params_sac["total_timesteps"] = int(1e6)
sac_params["learning_rate"] = 3e-4

env_args["tactile_sensor_params"]["type"] = "mini_right_angle_h_inner_tactip"
