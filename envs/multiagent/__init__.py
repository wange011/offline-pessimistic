import gym
from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

env_dict = gym.envs.registration.registry.env_specs.copy()


for env in env_dict:
    if 'MultiagentSimple-v0' in env or 'MultiagentSimpleSpeakerListener-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
