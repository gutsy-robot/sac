import datetime
import dateutil.tz
import os
import numpy as np
from rllab.spaces import Box

def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')

def concat_obs_z(obs, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[z] = 1
    return np.hstack([obs, z_one_hot])

def split_aug_obs(aug_obs, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return (obs, z)

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def _save_video(paths, filename):
    import cv2
    assert all(['ims' in path for path in paths])
    ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def _softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)

# wrapper to treat multi-agent env as single agent
class PseudoSingleAgentEnv():
    def __init__(self, env):
        self.env = env
        self.num_agents, self.single_obs_dim = np.array(self.env.reset()).shape
        self.observation_space = Box(low=-np.inf, high=-np.inf, shape=[self.num_agents*self.single_obs_dim])
        self.action_space = Box(low=-1, high=1, shape=[2*self.num_agents])
        self.spec = spec(self.observation_space, self.action_space)

    def reset(self):
        return np.array(self.env.reset()).flatten()
        
    def step(self, a):    # act_dim*num_agents
        a = np.split(a,self.num_agents)
        obs, reward, done, info = self.env.step(a)

        return np.array(obs).flatten(), np.sum(reward), np.any(done), info
    def log_diagnostics(paths, *args, **kwargs):
        print(paths)
class spec():
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))
