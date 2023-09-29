import numpy as np

rest_poses_dict = {
    "ur5": {
        "right_angle": {
            "main_robot":
                np.array(
                    [
                        0.00,  # world_joint            (fixed)
                        0.3459456028066371,  # base_joint         (revolute)
                        -1.6029931392619838,  # shoulder_joint     (revolute)
                        -2.5956992318566305,  # elbow_joint        (revolute)
                        -0.5154073337452144,  # wrist_1_joint      (revolute)
                        1.570570048554431,  # wrist_2_joint       (revolute)
                        -1.2247651630097756,  # wrist_3_joint      (revolute)
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
                        0.33584318461700613,  # base_joint         (revolute)
                        -1.6246857852046928,  # shoulder_joint     (revolute)
                        -2.5708422825829165,  # elbow_joint        (revolute)
                        -0.5186279940568668,  # wrist_1_joint      (revolute)
                        1.5705717484566102,  # wrist_2_joint       (revolute)
                        -1.2349526988037927,  # wrist_3_joint      (revolute)
                        0.00,  # ee_joint               (fixed)
                        0.00,  # tactip_ee_joint        (fixed)
                        0.00,  # tactip_body_to_adapter (fixed)
                        0.00,  # tactip_tip_to_body    (fixed)
                        0.00   # tcp_joint              (fixed)
                    ]
                ),
    }},
    "mg400": {
        "mini_right_angle_h_inner_tactip": {
            "main_robot":
                np.array(
                    [
                    -0.7633290833590991,     # j1        (revolute)
                    1.0463485970447421,     # j2_1         (revolute)
                    -0.5495075981644428,     # j3_1         (revolute)
                    -0.4979298780294844,     # j4_1          (revolute)
                    -0.8074701788897525,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.0463485970447421,      # j2_2 = j2_1         (revolute)
                    -1.0463485970447421 ,   # j3_2 = -j2_1         (revolute)
                    0.4968386553034963      # j4_2 = j2_1 + j3_1          (revolute)
                    ]
                ),

            "slave_robot":
                np.array(
                    [
                    -0.7809348456811154,     # j1        (revolute)
                    1.0057647239781813,     # j2_1         (revolute)
                    -0.4694053710791807,     # j3_1         (revolute)
                    -0.5374716584277155,     # j4_1          (revolute)
                    -0.7898636690928643,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_adaptor_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.0057648450323122,      # j2_2 = j2_1         (revolute)
                    -1.0057648450323122 ,   # j3_2 = -j2_1         (revolute)
                    0.5363566507277896      # j4_2 = j2_1 + j3_1          (revolute)
                    ]
                ),
                        },

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
                np.array([0,-0.35,0]),
            "base_rpy":
                np.array([0,0,np.pi/2]),
            # workframe 
            "update_init_pos":
                np.array([-0.10,0.0,0]),
            "update_init_rpy":
                np.array([0,0,0]),
                } ,
        "slave_robot": {
            "base_pos":
                np.array([0, 0.35, 0]),
            "base_rpy":
                np.array([0,0,-np.pi/2]),
            # workframe 
            "update_init_pos":
                np.array([0.10, 0, 0]),
            "update_init_rpy":
                np.array([0,0,np.pi]),
                } ,
            }, 
}