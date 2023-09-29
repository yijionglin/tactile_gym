import numpy as np

rest_poses_dict = {
    "ur5": {
        "right_angle": {
            "main_robot":
                np.array(
                    [
                        0.00,  # world_joint            (fixed)
                        -0.12558034380648742,  # base_joint         (revolute)
                        -1.7115553267527504,  # shoulder_joint     (revolute)
                        -2.5181744779327047,  # elbow_joint        (revolute)
                        -0.48395702311491073,  # wrist_1_joint      (revolute)
                        1.5698448713622477,  # wrist_2_joint       (revolute)
                        -1.6963863445956668,  # wrist_3_joint      (revolute)
                        0.00,  # ee_joint               (fixed)
                        0.00,  # tactip_ee_joint        (fixed)
                        0.00,  # tactip_body_to_adapter (fixed)
                        0.00,  # tactip_tip_to_body    (fixed)
                        0.00   # tcp_joint              (fixed)
                    ]
                ),

            "slave_robot":
                np.array(
                    [
                        0.00,  # world_joint            (fixed)
                        -0.12558034380648742,  # base_joint         (revolute)
                        -1.7115553267527504,  # shoulder_joint     (revolute)
                        -2.5181744779327047,  # elbow_joint        (revolute)
                        -0.48395702311491073,  # wrist_1_joint      (revolute)
                        1.5698448713622477,  # wrist_2_joint       (revolute)
                        -1.6963863445956668,  # wrist_3_joint      (revolute)
                        0.00,  # ee_joint               (fixed)
                        0.00,  # tactip_ee_joint        (fixed)
                        0.00,  # tactip_body_to_adapter (fixed)
                        0.00,  # tactip_tip_to_body    (fixed)
                        0.00   # tcp_joint              (fixed)
                    ]
                ),
    }},

    "mg400": {
        "mini_right_angle_tactip": {
            "main_robot":
                np.array(
                    [
                    0.8728163720612819,     # j1        (revolute)
                    1.0259855627669467,     # j2_1         (revolute)
                    -0.37267497119550547,     # j3_1         (revolute)
                    -0.6541928530735085,     # j4_1          (revolute)
                    -0.8721183569495453,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.0259855273595304,      # j2_2 = j2_1         (revolute)
                    -1.0259855273595304 ,   # j3_2 = -j2_1         (revolute)
                    0.6533127277493199      # j4_2 = j2_1 + j3_1          (revolute)
                    ]
                ),
            "slave_robot":
                np.array(
                    [
                    -0.8728162909480224,     # j1        (revolute)
                    1.0275594726827286,     # j2_1         (revolute)
                    -0.3758371981055423,     # j3_1         (revolute)
                    -0.652569142648328,     # j4_1          (revolute)
                    0.8721291910778106,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.0275594727293822,      # j2_2 = j2_1         (revolute)
                    -1.0275594727293822 ,   # j3_2 = -j2_1         (revolute)
                    0.6517246435758534      # j4_2 = j2_1 + j3_1          (revolute)
                    ]
                ),
                        }
                
        }
}
EEs_poses_sets = {
    "ur5": {
        "main_robot": {
            "base_pos":
                np.array([0,0,0]),
            "base_rpy":
                np.array([0,0,0]),
            "update_init_pos":
                np.array([0,0,0]),
            "update_init_rpy":
                np.array([0,0,0]),
                } ,
        "slave_robot": {
            "base_pos":
                np.array([0, 0.5, 0]),
            "base_rpy":
                np.array([0,0,0]),
            "update_init_pos":
                np.array([0, -0.5, 0]),
            "update_init_rpy":
                np.array([0,0,0]),
                } ,
            }, 
    "mg400": {
        "main_robot": {
            "base_pos":
                np.array([0,-0.40,0]),
            "base_rpy":
                np.array([0,0,np.pi/2]),
            # workframe 
            "update_init_pos":
                np.array([-0.25,0.15,0]),
            "update_init_rpy":
                np.array([0,0,0]),
                } ,
        "slave_robot": {
            "base_pos":
                np.array([0, 0.40, 0]),
            "base_rpy":
                np.array([0,0,-np.pi/2]),
            # workframe 
            "update_init_pos":
                np.array([-0.25, -0.15, 0]),
            "update_init_rpy":
                np.array([0,0,0]),
                } ,
            }, 
}