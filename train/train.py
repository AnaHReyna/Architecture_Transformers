import sys
sys.path.append('../')
import tensorflow as tf
import gym
from envs.carla.carla_env import InterSection
from init_configs import get_argument, set_configs
from algos.sac import SAC
from envs.runners.off_policy_trainer_carla import Trainer


args = get_argument().parse_args()

args, algo_params, runner_params = set_configs(args, test=False)


gpus = tf.config.experimental.list_physical_devices('GPU')
if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
    tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    print("Usando GPU:", gpus[args.gpu])
else:
    tf.config.set_visible_devices([], 'GPU')
    print("Treinando na CPU.")

    
OBSERVATION_SPACE = gym.spaces.Box(low=-1000, 
                                   high=1000, 
                                   shape=(args.neighbors + 1, 
                                    args.N_steps, 
                                    args.dim,))  

ACTION_SPACE = gym.spaces.Box(low=-1.0, 
                              high=1.0, 
                              shape=(2,))
    


env = InterSection(seed=0)
env.observation_space = OBSERVATION_SPACE
env.action_space = ACTION_SPACE


# args.model_dir = None

policy = SAC(state_shape=OBSERVATION_SPACE.shape, 
             action_dim=ACTION_SPACE.high.size, 
             max_action=ACTION_SPACE.high[0], 
             **algo_params)

runner = Trainer(policy=policy, 
                 env=env, 
                 args=args, 
                 test_env=env, 
                 **runner_params)
runner()