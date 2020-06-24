import os.path

from blox import AttrDict
from gcp.infra.agent.general_agent import GeneralAgent
from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import TopdownMultiroom3dEnv
from gcp.infra.policy.prm_policy.prm_policy import PrmPolicy

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'init_pos': None,
    'goal_pos': None,
    'n_rooms': 16,
    'heading_smoothing': 0.1,
    'crop_window': 40,
}

agent = AttrDict(
    type=GeneralAgent,
    env=(TopdownMultiroom3dEnv, env_params),
    T=100,
    make_final_gif=False,   # whether to make final gif
    #make_final_gif_freq=100,   # final gif, frequency
    image_height=128,
    image_width=128,
)

policy = AttrDict(
    type=PrmPolicy,
    max_traj_length=agent.T,
)

config = AttrDict(
    current_dir=current_dir,
    start_index=0,
    end_index=999,
    agent=agent,
    policy=policy,
    save_format=['hdf5'],
    data_save_dir=os.environ['GCP_DATA_DIR'] + '/nav_16rooms',
    split_train_val_test=False,
    traj_per_file=1,
)
