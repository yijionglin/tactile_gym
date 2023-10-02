# Tactile-Gym: RL suite for tactile robotics 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fac-93%2Ftactile_gym&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

---
<p align="center">
  <img width="800" src="docs/readme_images/BiTouch_cover.jpg"><br>
  This repository holds the reference implementations of three papers for RL-based tactile robotics.
</p>

---


**1) [Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10184426)** ([Project Website](https://sites.google.com/view/bi-touch/))

This work developed a dual-arm tactile robotic system (Bi-Touch) based on the [Tactile Gym 2.0](https://sites.google.com/view/tactile-gym-2/home) setup that integrates two affordable industrial-level robot arms (MG400) with low-cost high-resolution tactile sensors. A suite of bimanual manipulation tasks tailored toward tactile feedback is developed and open-sourced, indcluding bi-pushing, bi-gathering, bi-reorienting, and bi-lifting. We also successfully transferred the learned Bi-Touch policies into the real world, checkout our [project website](https://sites.google.com/view/bi-touch/) for more **sim-to-real demonstrations**.


<p align="center">
  <img width="180" src="docs/readme_videos/bipush.gif"> &nbsp;&nbsp;&nbsp;&nbsp;   
  <img  width="180" src="docs/readme_videos/bigather.gif"> &nbsp;&nbsp;&nbsp;&nbsp;   
  <img width="180" src="docs/readme_videos/bireorient.gif"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="180" src="docs/readme_videos/bilift.gif">  &nbsp;&nbsp;&nbsp;&nbsp;
</p>

---

**2) [Tactile Gym 2.0: Sim-to-real Deep Reinforcement Learning for Comparing Low-cost High-Resolution Robot Touch](https://ieeexplore.ieee.org/abstract/document/9847020)** ([Project Website](https://sites.google.com/view/tactile-gym-2/))

This updated version is built on top of the initial version [Tactile Gym 1.0](http://arxiv.org/abs/2106.08796) and now is open-sourced as [Tactile Gym 2.0](https://ieeexplore.ieee.org/abstract/document/9847020), which has been extended to include three tactile sensors (DigiTac, DIGIT, TacTip) of two widely-used yet fundamentally different types: [Tactip style](https://www.liebertpub.com/doi/full/10.1089/soro.2017.0052) and [Gelsight style](https://www.mdpi.com/1424-8220/17/12/2762). To make it easier for the research community of tactile robotics, we have also integrated a low-cost off-the-sheld industrial-level-accuracy desktop robot DOBOT MG400 for three learning environments as shown below, and successfully transferred the learned policies into the real world without any further fine-tuning (checkout our [project website](https://sites.google.com/view/tactile-gym-2/) for more sim-to-real demonstrations).


<p align="center">
  <img width="256" src="docs/readme_videos/sim_mg400_digitac_push.gif"> &nbsp;&nbsp;&nbsp;&nbsp;   
  <img  width="256" src="docs/readme_videos/sim_mg400_tactip_surf.gif"> &nbsp;&nbsp;&nbsp;&nbsp;   
  <img  width="256" src="docs/readme_videos/sim_mg400_digitac_surf.gif"> <br>
  <img width="256" src="docs/readme_videos/sim_ur5_digit_surf.gif"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="256" src="docs/readme_videos/sim_ur5_digitac_balance.gif">  &nbsp;&nbsp;&nbsp;&nbsp;
  <img width="256" src="docs/readme_videos/sim_mg400_digit_edge.gif"> &nbsp;&nbsp;&nbsp;&nbsp;
</p>

---

**3) [Tactile Gym 1.0: Tactile Sim-to-Real Policy Transfer via Real-to-Sim Image Translation](http://arxiv.org/abs/2106.08796)** ([Project Website](https://sites.google.com/view/tactile-gym-1/home))

This is the initial version that contains a suite of reinfocement learning environments built on top of [Tactile Sim](https://github.com/dexterousrobot/tactile_sim). These environments use tactile data as the main form of observations when solving tasks. This can be paired with [Tactile Sim2Real](https://github.com/dexterousrobot/tactile_sim2real) domain adaption for transferring learned policies to the real world.

<p align="center">
  <img width="256" src="docs/readme_videos/edge_follow.gif">
  <img width="256" src="docs/readme_videos/surface_follow.gif"> <br>
  <img width="256" src="docs/readme_videos/object_roll.gif">
  <img width="256" src="docs/readme_videos/object_push.gif">
  <img width="256" src="docs/readme_videos/object_balance.gif">
</p>



### Content ###
- [Installation](#installation)
- [Environment Details](#Environment-Details)
- [Observation Details](#observation-details)
- [Training Agents](#training-agents)
- [Re-training Agents](#re-training-agents)
- [Pretrained Agents](#pretrained-agents)
- [Alternate Robot Arms](#preliminary-support-for-alternate-robot-arms)
- [Bibtex](#bibtex)


### Installation ###
This repo has only been developed and tested with Ubuntu 18.04 and python 3.8.

```console
git clone https://github.com/dexterousrobot/tactile_gym
cd tactile_gym
pip install -e .
```

Demonstration files are provided in the example directory. From the base directory run

```
python examples/demo_env.py -env example_arm-v0
```

alternate envs can be specified but setting the `-env` argurment to any of the following: `example_arm-v0`, `edge_follow-v0`, `surface_follow-v0`, `object_roll-v0`, `object_push-v0`, `object_balance-v0`, `bipush-v0`, `bigather-v0`, `bireorient-v0`, `bilift-v0`.

Usage: You can specify a desired robot arm and a tactile sensor and other environment parameters within the `demo_env.py` file.

### Environment Details ###

| **Env. Name** | Description |
| :---: | :--- |
| `bipush-v0` | <ul><li>A large-size object is placed within the environment. </li><li>The sensors are initialised to contact random levels of pentration at the start of the object.</li><li>The objective is to move the large-size object on a planar surface collaboratively with two robot arms with tactile feadback to achieve a sequence of goals along a given trajectory.</li></ul>  |
| `bigather-v0` | <ul><li>Two objects are placed within the environment. </li><li>The sensors are initialised to contact random levels of pentration at the start of the objects</li><li>The objective is to gather two objects by pushing them towards each other on a planar surface under random perturbations.</li></ul>  |
| `bireorient-v0` | <ul><li>An object with randomized size is placed at the workspace centre within the environment. </li><li>The sensors are initialised to contact random levels of pentration at the start of the object.</li><li>The objective is to reorient the object to a given target angle while keeping the object centre fixed in place.</li></ul>  |
| `bilift-v0` | <ul><li>An object with randomized size is placed at the workspace centre within the environment. </li><li>The sensors are initialised to contact random levels of pentration at the start of the object.</li><li>The objective is to gently lift the object and keep it along a given trajectory in the air to achieve a sequence of goals.</li></ul>  |
| `edge_follow-v0` | <ul><li>A flat edge is randomly orientated through 360 degrees and placed within the environment. </li><li>The sensor is initialised to contact a random level of pentration at the start of the edge.</li><li>The objective is to traverse the edge to a goal at the oposing end whilst maintaining that the edge is located centrally on the sensor.</li></ul>  |
| `surface_follow-v0`   | <ul><li>A terrain like surface is generated through [OpenSimplex Noise](https://pypi.org/project/opensimplex/).</li><li>The sensor is initialised in the center, touching the surface.</li><li>A goal is randomly placed towards the edges of the surface.</li><li>The objective is to maintain a normal orientation to the surface and a set penetration distance whilst the sensor is automatically moved towards the goal.</li></ul> |
| `surface_follow-v1`   | <ul><li>Same as `-v0` however the goal location is included in the observation and the agent must additionally learn to traverse towards the goal.</li></ul> |
| `surface_follow-v2`   | <ul><li>Same as `-v0` but verticalized, and this is used for training 4-DoF robots (MG400, Magician) which only have x-, y-, z-, and Rz-axes actuated.</li></ul> |
| `object_roll-v0`   | <ul><li>A small spherical object of random size is placed on the table.</li><li>A flat tactile sensor is initialised to touch the object at a random location relative to the sensor.</li><li>A goal location is generated in the sensor frame.</li><li>The objective is to manipulate the object to the goal location.</li></ul> |
| `object_push-v0`   | <ul><li>A cube object is placed on the table and the sensor is initialised to touch the object (in a right-angle configuration).</li><li>A trajectory of points is generated through OpenSimplex Noise.</li><li>The objective is to push the object along the trajectory, when the current target point has been reached it is incremented along the trajectory until no points are left.</li></ul> |
| `object_balance-v0`   | <ul><li>Similar to a 2d CartPole environment.</li><li>An unstable pole object is balanced on the tip of a sensor pointing upwards.</li><li>A random force pertubation is applied to the object to cause instability.</li><li>The objective is to learn planar actions to counteract the rotation of the object and mantain its balanced position.</li></ul> |

### Tactile Sensors ###

| **Sensor Name** | Description |
| :---: | :--- |
| `TacTip` | [TacTip](https://www.liebertpub.com/doi/full/10.1089/soro.2017.0052) is a soft, curved, 3D-printed tactile skin with an internal array of pins tipped with markers, which are used to amplify the surface deformation from physical contact against a stimulus.  |
| `DIGIT`   |  [DIGIT](https://digit.ml/) shares the same principle of the [Gelsight tactile sensor](https://www.mdpi.com/1424-8220/17/12/2762), but can be fabricated at low cost and is of a size suitable for integration of some robotic hands, such as on the fingertips of the Allegro. |
| `DigiTac`   | [DigiTac](https://lepora.com/digitac/) is an adapted version of the DIGIT and the TacTip, whereby the 3D-printed skin of a TacTip is customized to integrated onto the DIGIT housing, while keeping the camera and lighting system. In other words, this sensor outputs tactile images of the same dimension as the DIGIT, but with a soft biomimetic skin like other TacTip sensors. |



### Observation Details ###

All environments contain 4 main modes of observation:

| Observation Type | Description |
| :---: | :--- |
| `oracle` | Comprises ideal state information from the simulator, which is difficult information to collect in the real world, we use this to give baseline performance for a task. The information in this state varies between environments but commonly includes TCP pose, TCP velocity, goal locations and the current state of the environment. This observation requires signifcantly less compute both to generate data and for training agent networks.|
| `tactile` | Comprises images (default 128x128) retrieved from the simulated optical tactile sensor attached to the end effector of the robot arm (Env Figures right). Where tactile information alone is not sufficient to solve a task, this observation can be extended with oracle information retrieved from the simulator. This should only include information that could be be easily and accurately captured in the real world, such as the TCP pose that is available on industrial robotic arms and the goal pose. |
| `visual` | Comprises RGB images (default 128x128) retrieved from a static, simulated camera viewing the environment (Env Figures left). Currently, only a single camera is used, although this could be extended to multiple cameras. |
| `visuotactile` |  Combines the RGB visual and tactile image observations to into a 4-channel RGBT image. This case demonstrates a simple method of multi-modal sensing. |

When additional information is required to solve a task, such as goal locations, appending `_and_feature` to the observation name will return the complete observation.


### Training Agents ###

The environments use the [OpenAI Gym](https://gym.openai.com/) interface so should be compatible with most reinforcement learning librarys.

We use [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for all training, helper scripts are provided in `tactile_gym/sb3_helpers/`

A simple experiment can be run with `simple_sb3_example.py`, a full training script can be run with `train_agent.py`. Experiment hyper-params are in the `parameters` directory.

### Re-training Agents ###

Now you can retrain your model by:
```bash
python train_agent.py --if_retrain=True --retrain_path=~/saved_models/env_name/algo_name/
```

This can help you quickly train a policy that has already been learned with one robot for another robot using the same tactile sensor, instead of starting from scratch.

### Pretrained Agents ###
RL models for DIGIT and DigiTac: these can be downloaded [here](https://drive.google.com/drive/folders/1VVrnMVaQeX05C9Uo4126ATDNIPFpf2Ga?usp=sharing)
and placed in `tactile_gym/sb3_helpers/saved_models`.

```
python tactile_gym/sb3_helpers/eval_agent_utils.py
```

Example PPO/RAD_PPO agents, trained via SB3 are provided for all environments and all observation spaces. These can be downloaded [here](https://drive.google.com/drive/folders/1stIhPc0HBN8fcJfMq6e-wHcsp6VpJafQ?usp=sharing)
and placed in `tactile_gym/examples/enjoy`.

In order to demonstrate a pretrained agent from the base directory run
```
python examples/demo_trained_agent.py -env='env_name' -obs='obs_type' -algo='algo_name'
```

### Preliminary Support for Alternate Robot Arms ###

The majority of testing is done on the simulated UR5 and MG400 robot arm. The Franka Emika Panda and Kuka LBR iiwa robot arms are additionally provided however there may be bugs when using these arms. Particularly, workframes may need to be adjusted to ensure that arms can comfortably reach all the neccessary configurations. These arms can be used by changing the `robot_arm_params.type` flag within the code in the params folder.

<p align="center">
  <img src="docs/readme_videos/surf_arm_transfer.gif">
</p>

### Bibtex ###

If you find it useful for your research, please cite our papers:

```
@ARTICLE{lin2023bitouch,
  author={Lin, Yijiong and Church, Alex and Yang, Max and Li, Haoran and Lloyd, John and Zhang, Dandan and Lepora, Nathan F.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Bi-Touch: Bimanual Tactile Manipulation With Sim-to-Real Deep Reinforcement Learning}, 
  year={2023},
  volume={8},
  number={9},
  pages={5472-5479},
  doi={10.1109/LRA.2023.3295991},
  url={https://ieeexplore.ieee.org/abstract/document/10184426},
  }


@ARTICLE{lin2022tactilegym2,
  author={Lin, Yijiong and Lloyd, John and Church, Alex and Lepora, Nathan F.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Tactile Gym 2.0: Sim-to-Real Deep Reinforcement Learning for Comparing Low-Cost High-Resolution Robot Touch}, 
  year={2022},
  volume={7},
  number={4},
  pages={10754-10761},
  doi={10.1109/LRA.2022.3195195},
  url={https://ieeexplore.ieee.org/abstract/document/9847020},
  }


@InProceedings{church2021optical,
     title={Tactile Sim-to-Real Policy Transfer via Real-to-Sim Image Translation},
     author={Church, Alex and Lloyd, John and Hadsell, Raia and Lepora, Nathan F.},
     booktitle={Proceedings of the 5th Conference on Robot Learning},
     year={2022},
     editor={Faust, Aleksandra and Hsu, David and Neumann, Gerhard},
     volume={164},
     series={Proceedings of Machine Learning Research},
     month={08--11 Nov},
     publisher={PMLR},
     pdf={https://proceedings.mlr.press/v164/church22a/church22a.pdf},
     url={https://proceedings.mlr.press/v164/church22a.html},
}
```
