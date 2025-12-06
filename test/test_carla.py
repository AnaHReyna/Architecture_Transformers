import sys
sys.path.append('../')
import tensorflow as tf
import gym
from envs.carla.carla_env import InterSection
from configs.init_configs import get_argument, set_configs      
from envs.runners import on_policy_trainer_carla, off_policy_trainer_carla




args = get_argument().parse_args()
args, algo_params, runner_params = set_configs(args, test=True)



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
    tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    print(gpus[args.gpu])
else:
    # força CPU
    tf.config.set_visible_devices([], 'GPU')



# if args.scenario =='carla':
    # carla env spec
OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim,))
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
test_env = InterSection()
test_env.observation_space = OBSERVATION_SPACE
test_env.action_space = ACTION_SPACE

if args.model_dir == None:
    args.model_dir = f'../data/{args.algo}/{args.scenario}/ckpt'
    print("Usando modelo padrão em:", args.model_dir)
    # args.model_dir = f'../train_exp/results/{args.algo}/ckpt-5'


if args.algo == 'ppo':
    from algos.modules.actor_critic_policy import GaussianActorCritic
    from algos.ppo import PPO
    policy = PPO(actor_critic=GaussianActorCritic(**algo_params['actor_critic_params']), **algo_params)
    runner_page = on_policy_trainer_carla
    runner = runner_page.OnPolicyTrainer(policy=policy, env=test_env, args=args, test_env=test_env, **runner_params)
    runner.evaluate_policy()
else:
    from algos.sac import SAC
    policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], **algo_params)
    runner_page = off_policy_trainer_carla
    runner = runner_page.Trainer(policy=policy, env=test_env, test_env=test_env, args=args, **runner_params)
    runner.evaluate_policy_continuously()








# print(OBSERVATION_SPACE.shape,algo_params)
# if args.algo == 'ppo':
#     policy = SAC(max_action=ACTION_SPACE.high[0], **algo_params)
# else:
#     policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], **algo_params)
# runner = Trainer(policy=policy, env=test_env, args=args, **runner_params)
# runner.evaluate_policy_continuously() 