# [RAL 2025] Following Is All You Need: Robot Crowd Navigation Using People As Planners

## Getting started
This code has been tested on [ROS1](https://wiki.ros.org/ROS/Tutorials) Noetic with Python 3.8. 
We provide the complete environment on Docker:
```
docker pull xuxinhang007/open_source:people_as_planner

docker run -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/iot/docker_mount:/root/mnt \
  -e DISPLAY=unix$DISPLAY \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name people_as_planner \
  xuxinhang007/open_source:people_as_planner
```

This repo contains the main implementation code for our method and the baselines (excluding the Gazebo setup, which is only available in our Docker environment). 
The main code for our method PeopleAsPlanner and the parameters can be found in `our_robot.py`.
The baselines and other dependencies are marked by submodules.

You may build upon this code to work with other simulators or real-world robots.

## Experiments
After building and sourcing the workspace `sim_ws`, you may run an experiment by (our method on the Crossing scene, for example):
```
roslaunch data_play eval_our.launch
```
There is a separate launch file for each method, and the scene is set in the launch file by:
```
<arg name="scene" default="crossing_0"/> <!-- Edit scene here -->
```
It may take a while for the Rviz and Gazebo windows to spawn, and the experiment data will be recorded in a new `test_data` folder. 

After multiple runs, you may use `matric_cal_new.py` to calculate the metrics for each run, and `from_txt_to_table.py` to get a summary of all runs.

Feel free to raise an issue if you have any questions :blush:

## TODO
- [ ] update RAL citation

## Citation
If you find this work useful, please cite [Following Is All You Need: Robot Crowd Navigation Using People As Planners](https://arxiv.org/abs/2504.10828) ([pdf](https://arxiv.org/abs/2504.10828), [video](https://youtu.be/xnX6_-D2ZfQ)):

```bibtex
@misc{liao2025followingneedrobotcrowd,
      title={Following Is All You Need: Robot Crowd Navigation Using People As Planners}, 
      author={Yuwen Liao and Xinhang Xu and Ruofei Bai and Yizhuo Yang and Muqing Cao and Shenghai Yuan and Lihua Xie},
      year={2025},
      eprint={2504.10828},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.10828}, 
}
```

## Contributors
<a href="https://github.com/centiLinda/PeopleAsPlanner/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=centiLinda/PeopleAsPlanner" />
</a>