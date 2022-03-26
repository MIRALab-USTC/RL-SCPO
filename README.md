# Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization
This is the code of paper **Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization**. [[arXiv]](https://arxiv.org/abs/2112.10513)


### Instructions

- install requirements (python=3.6):

```
pip install -r requirements.txt
```

- run sc-sac in the Walker2d-v2 task with default configs:

```
python launch.py -e Walker2d-v2 
```

- run sac in the HalfCheetah-v2 task:

```
python launch.py -n sac -e HalfCheetah-v2 
```

- run pr-sac in the Hopper-v2 task:

```
python launch.py -n pr-sac -e Hopper-v2
```

- plot heatmap of sc-sac trained policies in the HalfCheetah-v2 task:

```
python plot.py HalfCheetah-v2 sc-sac /path/to/data/save/dir
```

- note that before ploting the heatmap, you have to manually replace the codes in /path_to_gym_module/envs/mujoco with the codes in ./mujoco_env_enhancing_codes, which enables us to change the relative mass and the relative friction of these environments during test.

```
cp -r ./mujoco_env_enhancing_codes/* /path_to_gym_module/envs/mujoco/
```