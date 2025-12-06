import argparse
import yaml
import sys
sys.path.append('../')


def get_argument(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(conflict_handler='resolve')
    # experiment settings
    parser.add_argument('--max-steps', type=int, default=100_000,
                        help='Maximum number steps to interact with env.')
    parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                        help='Maximum steps in an episode')
    parser.add_argument('--n-experiments', type=int, default=3,
                        help='Number of experiments')
    parser.add_argument('--show-progress', action='store_true',
                        help='Call `render` in training process')
    parser.add_argument('--save-model-interval', type=int, default=20000,
                        help='Interval to save model')
    parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                        help='Interval to save summary')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory to restore model')    # para teste ou restauro de treino (model-dir)
    parser.add_argument('--dir-suffix', type=str, default='',
                        help='Suffix for directory that contains results')
    parser.add_argument('--normalize-obs', action='store_true', default=True,
                        help='Normalize observation')
    parser.add_argument('--logdir', type=str, default='results',
                        help='Output directory')
    # test settings
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model')
    parser.add_argument('--test-interval', type=int, default=int(20e4),
                        help='Interval to evaluate trained model')
    parser.add_argument('--show-test-progress', action='store_true',
                        help='Call `render` in evaluation process')
    parser.add_argument('--test-episodes', type=int, default=20,
                        help='Number of episodes to evaluate at once')
    parser.add_argument('--save-test-path', action='store_true',
                        help='Save trajectories of evaluation')
    parser.add_argument('--show-test-images', action='store_true',
                        help='Show input images to neural networks when an episode finishes')
    parser.add_argument('--save-test-movie', action='store_true',
                        help='Save rendering results')
    # replay buffer
    parser.add_argument('--use-prioritized-rb', action='store_true', default=False,
                        help='Flag to use prioritized experience replay')
    
    parser.add_argument('--use-nstep-rb', action='store_true', default=False,
                        help='Flag to use nstep experience replay')
    parser.add_argument('--n-step', type=int, default=4,
                        help='Number of steps to look over')
    # others
    parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                        default='INFO', help='Logging level')
    parser.add_argument('--scenario', choices=['left_turn', 'cross', 'carla', 'roundabout',
                        'roundabout_easy', 'roundabout_medium'], default='carla')
    parser.add_argument('--algo',choices=['scene_rep', 'drq', 'ppo', 'dt', 'rb'], default='scene_rep')

    #state settings     
    parser.add_argument('--N-steps', type=int, default=10, help='Number of history steps for logging, 3 for image and 10 for vector')
    parser.add_argument('--planned-path', type=int, default=10, help='length for map segments')
    parser.add_argument('--neighbors', type=int, default=5, help='num of closest neighbors')          
    parser.add_argument('--dim', type=int, default=5, help='dims for state features') 
    parser.add_argument('--head-nums', type=int, default=2, help='nums of MHSA')             
    parser.add_argument('--future-steps', type=int, default=3, help='Number of future steps for predictions')

    #learning settings:
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpus')

    # runner settings
    parser.add_argument('--skip-timestep', type=int, default=3, help='time step interval for actions')
    return parser


def set_configs(args, test=False):
    return set_off_policy_configs(args, test)
    

def set_off_policy_configs(args, test=False):
    algo = args.algo
    scenario = args.scenario
    encoding_params = {
        'bptt':False,
        'ego_surr':False,
        'cnn': False,
        'neighbours':args.neighbors,
        'time_step':args.N_steps,
        'make_rotation':True,
        'make_prediction_q':False,
        'make_prediction_value':False,        
        'num_traj':1,        
        'traj_length':args.future_steps,
        'use_map':True,
        'state_input':False,
        'LSTM':False,
        'cnn_lstm':False,
        'path_length':args.planned_path,
        'head_num':args.head_nums,
        'n_steps':args.n_step,
        'N_steps':args.N_steps,
        'future_step':args.future_steps,
        'use_hier':True,
        'random_aug':True,
        'no_ego_fut':False,
        'no_neighbor_fut':False,
        'carla': (scenario =='carla'),
        'units': 128
    }

    if scenario in ['carla', 'cross']:
        encoding_params['head_num'] = 2
    

    representations = (algo =='scene_rep')  # True or False   # classe Represent_Learner em actor_critic.py
    use_map = True


    if test:
        representations = False
        encoding_params['random_aug'] = False
    
    # print('params for RL encoder:', encoding_params)

    algo_params = dict(
        params=encoding_params,
        batch_size=args.batch_size, 
        auto_alpha=True, 
        memory_capacity=int(2e4),
        n_warmup=5000, 
        lr=args.lr, #3e-4 for Drq
        gpu=args.gpu,
        discount=args.discount,
        actor_critic=True,
        representations=representations,
        sep_rep_opt=representations,
        aug=(algo=='drq'),
    )

    runner_params = dict(
        params=encoding_params,
        make_predictions=args.future_steps if representations else 0, 
        use_map=use_map,
        path_length=args.planned_path,
        neighbors=args.neighbors, 
        pred_future_state=representations,
        sep_train=representations,
        skip_timestep=args.skip_timestep,
    )

    return args, algo_params, runner_params