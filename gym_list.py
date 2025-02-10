import gym

envs = gym.envs.registry
env_list = list(envs.keys())

for env in sorted(env_list):
    print(env)
