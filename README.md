# Hierarchies-of-Planning-and-Reinforcement-Learning-for-Robot-Navigation

This is the official code, implemented by Jan Wöhlke, accompanying the ICRA 2021 paper Hierarchies of Planning and Reinforcement Learning 
for Robot Navigation by Jan Wöhlke, Felix Schmitt, and Herke van Hoof. The paper can be found [here](https://arxiv.org/abs/2109.11178). The code allows 
the users to reproduce and extend the results reported in the paper. Please 
cite the above paper when reporting, reproducing or extending the results:
```
@inproceedings{woehlke2021hierarchies,
title = {Hierarchies of Planning and Reinforcement Learning for Robot Navigation},
author = {Jan W{\"o}hlke and Felix Schmitt and van Hoof, Herke},
year = {2021},
booktitle = {IEEE International Conference on Robotics and Automation}
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor 
monitored in any way.

## Requirements, how to build, test, install, use, etc.

Place the folder [code](code) where you want to execute the code.

Add the path of the folder [code](code) to your $PYTHONPATH environment variable.


#### Simple Point-Mass Navigation

You need a Python set-up with the following packages:
* python>=3.7.6
* numpy>=1.18.1
* scipy>=1.4.1
* torch==1.3.1

For the "Four Rooms" simple point-mass navigation experiments in Section V.C. / 
Figure 4 a) and b) in the paper you find the start scripts [here](scripts/4_rooms):

Without "terrain":
* [BSL](scripts/4_rooms/bsl_trpo_4_rooms.py)
* [HIRO](scripts/4_rooms/hiro_4_rooms.py)
* [HIRO HER](scripts/4_rooms/hiro_her_4_rooms.py)
* [MVPROP-RL](scripts/4_rooms/mvprop_rl_4_rooms.py)
* [RRT-WP](scripts/4_rooms/rrt_wp_4_rooms.py)
* [VI-RL](scripts/4_rooms/vi_rl_4_rooms.py)
* [VI-RL OM](scripts/4_rooms/vi_rl_om_4_rooms.py)

With "terrain":
* [BSL](scripts/4_rooms/bsl_trpo_4_rooms_t.py)
* [HIRO](scripts/4_rooms/hiro_4_rooms_t.py)
* [HIRO HER](scripts/4_rooms/hiro_her_4_rooms_t.py)
* [MVPROP-RL](scripts/4_rooms/mvprop_rl_4_rooms_t.py)
* [RRT-WP](scripts/4_rooms/rrt_wp_4_rooms_t.py)
* [VI-RL](scripts/4_rooms/vi_rl_4_rooms_t.py)
* [VI-RL OM](scripts/4_rooms/vi_rl_om_4_rooms_t.py)

Run the scripts with: _python [scriptname] --seed=[seed option]_

There are different seed options to run the number random seeds used for 
the respective experiment:
* "s1" to "s10" : run an individual random seed
* not specified / other: run all ten seeds in sequence


#### Robotic Maze Navigation

You need to have [Mujoco](http://www.mujoco.org/) (version 2.0 - mujoco/200) installed and have a valid license key for it.

You need to install the following additional Python packages:
* gym==0.15.4 (install in editable mode: pip install -e)
* mujoco-py (our version: 2.0.2.10, https://github.com/openai/mujoco-py)

The start scripts for the experiments in Section V.D. / Figure 4 c) can be found [here](scripts/ant_44):
* [BSL](scripts/ant_44/bsl_trpo_mj_ant_44.py)
* [MVPROP-RL](scripts/ant_44/mvprop_rl_trpo_ant_44.py)
* [RRT-WP](scripts/ant_44/rrt_wp_trpo_mj_ant_44.py)
* [VI-RL](scripts/ant_44/vi_rl_trpo_mj_ant_44.py)
* [VI-RL OM](scripts/ant_44/vi_rl_om_trpo_mj_ant_44.py)

Before running the scripts, you need to fill the folder path of the _assets_ folder, where the
_ant.xml_ file is located, of your _gym_ installation as the _mj_ant_path_ (where it says 
_'TO_BE_ENTERED'_ in the respective script).

Note that during execution the _ant.xml_ will be modified constantly. In case the program does
not terminate properly, it might remain in a modified state and needs to be restored from the 
generated _ant_copy.xml_ file, in order to obtain correct results in subsequent simulations. 

Run the scripts with: _python [scriptname] --seed=[seed option]_

There are different seed options to run the number random seeds used for 
the respective experiment:
* "s1" to "s10" : run an individual random seed
* not specified: run all ten seeds in sequence


#### Non-Holonomic Vehicle Parking

You need to install the following additional Python packages:
* gym==0.15.4 (may already be done for the robotic maze navigation experiments)
* highway-env (https://github.com/eleurent/highway-env)

The start scripts for the experiments in Section V.E. / Figure 5 can 
be found [here](scripts/parking):
* [BSL](scripts/parking/bsl_trpo_parking.py)
* [HIRO HER](scripts/parking/hiro_parking.py)
* [MVPROP-RL](scripts/parking/mvprop_rl_parking.py)
* [VI-RL](scripts/parking/vi_rl_parking.py)
* [VI-RL OM](scripts/parking/vi_rl_om_parking.py)

Run the scripts with: _python [scriptname] --seed=[seed option]_

There are different seed options to run the number random seeds used for 
the respective experiment:
* "s1" to "s10" : run an individual random seed
* not specified: run all ten seeds in sequence


#### RESULTS

Plot the test goal reaching probabilities in the "results_ ..." files of 
all the random seeds of the respective experiment of interest as mean +- 
standard error to arrive at the curves shown in the paper.

## License

Hierarchies-of-Planning-and-Reinforcement-Learning-for-Robot-Navigation 
is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) 
file for details.

For a list of other open source components included in 
Hierarchies-of-Planning-and-Reinforcement-Learning-for-Robot-Navigation, 
see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).
