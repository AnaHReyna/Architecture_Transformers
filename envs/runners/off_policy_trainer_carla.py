import os, re
import time
import logging
import argparse
import csv
import json
import pickle

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from copy import deepcopy

import random
from matplotlib import pyplot as plt
from time import sleep
from scipy.stats import norm
from tqdm import tqdm

from .get_rb import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tensorflow.keras.models import load_model
from collections import deque



class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None,
            use_mask=True,
            bptt_hidden=0,
            make_predictions=10,
            path_length = 10,
            use_map=True,
            params=None,     # parmetros das camadas de representação
            neighbors=5,
            multi_selection=False,
            pred_future_state=True,
            skip_timestep=3,
            sep_train=True
        ):

        
        self.return_log = []
        self.eval_log = []
        self.step_log = []
        self.test_step = []

        self.train_success_rate=[]

        self.test_success_rate=[]

        self.use_mask=use_mask
        self.timesteps=0
        self.bptt_hidden=bptt_hidden
        self.use_map = use_map    
        self.path_length=path_length 

        self.params = params
        self.sep_train = sep_train

        self.make_predictions = make_predictions
        self.multi_selection = multi_selection
        self.neighbors = neighbors

        self.pred_future_state = pred_future_state
        self.skip_timestep = skip_timestep

        if isinstance(args, dict):
            _args = args   
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([]) 
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env 
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)
        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))

        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)


        # if evaluate the model
        if self.use_mask:
            self.timesteps=10
            # if self.timesteps==80:
                # self.timesteps=int(self._env.observation_space.shape[-1]/3)
            print('using mask,timesteps:',self.timesteps)
        if self.bptt_hidden>0:
            print('using bptt,hidden size',self.bptt_hidden)
        if self.make_predictions>0:
            print('use predictions')
            if self.multi_selection:
                print('using multi selection...')
            if self.pred_future_state:
                print('representation learning..')

        if self.use_map:
            print('use_ego neighbor maps')
          

        # if args.evaluate:   # Não precisa disso
            # assert args.model_dir is not None
            # if args.model_dir is None:
                # raise AssertionError
            
        self._set_check_point(args.model_dir) # restaura o modelo se args.model_dir não for None

        self.step_checkpoint = self._ckpt_step_from_path(args.model_dir)  # obtém o número do passo do checkpoint se args.model_dir for um arquivo de checkpoint



        # prepare TensorBoard output
        self.train_logdir = os.path.join(self._output_dir, "tb", "train")
        self.eval_root    = self._run_dir_from_model_dir(args.model_dir)  # avaliações devem ir para a pasta do experimento (a do ckpt, se for prefixo)
        self.eval_logdir  = os.path.join(self.eval_root, "tb", "eval")

        self.tb_train = tf.summary.create_file_writer(self.train_logdir)
        # self.tb_train.set_as_default()

        self.tb_eval = tf.summary.create_file_writer(self.eval_logdir)
        # self.tb_eval.set_as_default()

        # self.writer = tf.summary.create_file_writer(self._output_dir)
        # self.writer.set_as_default()


    def _run_dir_from_model_dir(self, md):
        if md and not os.path.isdir(md):
            return os.path.dirname(os.path.abspath(md)) # arquivo de checkpoint
        return self._output_dir
    
    def _ckpt_step_from_path(self, p):
        if not p:
            return None
        m = re.search(r'ckpt-(\d+)', str(p))    
        return int(m.group(1)) if m else None   
        
    

    def _adapt_map_state(self, ms, neighbors=5, path_len=10, target_dim=5):
        """
        Converte CARLA map_s (18,10,2) -> (12,10,5)
        Mantém 2 rotas por agente; faz pad/trunc das features até target_dim.
        """
        if ms is None:
            return None
        ms = np.asarray(ms)
        if ms.ndim != 3 or ms.shape[1] != path_len:
            return ms  # já está no formato esperado

        n_agents = neighbors + 1 
        K, L, F = ms.shape  # (K=18, L=10, F=2)
        
        if K % n_agents == 0:
            n_heads = K // n_agents  # ex.: 3
        else:
            n_heads = 2

        # reshape para (agentes, rotas, L, F)
        try:
            ms = ms.reshape(n_agents, n_heads, L, F)
        except Exception:
            # se não der, tenta truncar/duplicar K para encaixar
            want = n_agents * max(2, n_heads)
            if K < want:
                reps = int(np.ceil(want / K))
                ms = np.tile(ms, (reps, 1, 1))[:want]
            else:
                ms = ms[:want]
            n_heads = want // n_agents
            ms = ms.reshape(n_agents, n_heads, L, F)

        # escolhe 2 rotas por agente: use as laterais [0, -1] (descarta a central)
        if n_heads >= 2:
            ms = ms[:, (0, -1), :, :]          # (agentes, 2, L, F)
        else:
            # se só há 1, duplique
            ms = np.repeat(ms, 2, axis=1)

        # volta para (agentes*2, L, F) -> ex.: (12,10,2)
        ms = ms.reshape(n_agents * 2, L, F)

        # pad/trunc de features para target_dim (ex.: 5)
        f = ms.shape[-1]
        if f < target_dim:
            pad = np.zeros(ms.shape[:-1] + (target_dim - f,), dtype=ms.dtype)
            ms = np.concatenate([ms, pad], axis=-1)
        elif f > target_dim:
            ms = ms[..., :target_dim]  # trunc # 

        return ms


    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy) # objeto de checkpoint
        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, directory=self._output_dir, max_to_keep=5) # Gerenciador de checkpoints

        if model_dir is not None:
            if not os.path.isdir(model_dir):  # os.path.isdir(model_dir) se model_dir é um diretório e existe o caminho
                self._latest_path_ckpt = model_dir    # Aqui model_dir é um arquivo de checkpoint
            else:
                self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt).expect_partial()
            self.logger.info("Restored {}".format(self._latest_path_ckpt))


    def __call__(self):
        total_steps = 0
        frame_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        episode_returns = []
        success_log = [0]
        best_train = -np.inf

        replay_buffer = get_replay_buffer(
                            self._policy, self._env, self._use_prioritized_rb, self._use_nstep_rb, 
                            self._n_step,self._policy.memory_capacity, timesteps=self.timesteps, 
                            bptt_hidden=self.bptt_hidden, make_predictions=self.make_predictions,
                            use_map=self.use_map,path_length=self.path_length,neighbors=self.neighbors,
                            multi_selection=self.multi_selection,represent=self.pred_future_state)
        
        if self.make_predictions>0:
            local_queue = deque(maxlen=self.make_predictions)
            ego_queue = deque(maxlen=self.make_predictions)
        
            if self.pred_future_state:
                fut_state_queue = deque(maxlen=self.make_predictions)
                fut_action_queue = deque(maxlen=self.make_predictions)
                if self.use_map:
                    fut_map_queue = deque(maxlen=self.make_predictions)


        obs = self._env.reset()
        reset_info = None
        if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[1], dict):
            obs, reset_info = obs

        if isinstance(reset_info, dict):
            vision = reset_info.get("vision", None)
        else:
            vision = None
        next_vision = None

        if self.use_map:
            obs, ego, map_s = obs   # neighbor_trajs, ego_traj[-1], neighbor_waypoints[:,::5]
            map_s = self._adapt_map_state(map_s, neighbors=self.neighbors, path_len=self.path_length, target_dim=5)


        if self.bptt_hidden>0:
            hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
        else:
            hidden,full_hidden,next_hidden=None, None, None
        r = 0
        b_s = 0
        dis = 0
        while total_steps < self._max_steps:

            if self.use_mask:
                num = np.clip(episode_steps+1,0,self.timesteps)
                mask = np.array([1]*num +[0]*(self.timesteps - num))
                mask = np.expand_dims(mask, axis=0)

                n_num = np.clip(episode_steps+2,0,self.timesteps)
                next_mask = np.array([1]*n_num +[0]*(self.timesteps - n_num))
                next_mask = np.expand_dims(next_mask, axis=0)
            else:
                mask=None
                next_mask=None


            pred_ego = None   # Atualizei aqui
            if episode_steps%self.skip_timestep==0:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                else:
                    action = self._policy.get_action(obs,mask=mask,map_state=np.expand_dims(map_s,axis=0),test=False)
                    # pred_ego = None   # Aqui comentei
                act = action

            next_obs, reward, done, info = self._env.step(act)

            if isinstance(info, dict):
                next_vision = info.get("vision", None)
            else:
                next_vision = None


            if self.use_map:
                next_obs, next_ego, next_map_s = next_obs
                next_map_s = self._adapt_map_state(next_map_s, neighbors=self.neighbors, path_len=self.path_length, target_dim=5)

            
            r += self._policy.discount**(b_s) * reward
            b_s +=1

            episode_steps += 1
            episode_return += reward
            total_steps += 1
            # tf.summary.experimental.set_step(total_steps)

            # if the episode is finished
            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                episode_steps == self._env._max_episode_steps):
                done_flag = False
            
            if (episode_steps%self.skip_timestep==0) or done or episode_steps == self._episode_max_steps:
                reward = r
                r = 0
                b_s = 0
                frame_steps+= 1
            
            if self.bptt_hidden>0 :
                if total_steps > self._policy.n_warmup:
                    next_hidden = np.expand_dims(full_hidden[(episode_steps-1)%self.timesteps],0)
                    # print(hidden.shape)
                    if episode_steps%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)
                else:
                    next_hidden = hidden
            # print(episode_steps)
            if episode_steps%self.skip_timestep==0 or done or episode_steps == self._episode_max_steps:
                
                if self.make_predictions>0:

                    line = [obs, action, next_obs, reward, done_flag, mask, hidden, next_mask, next_hidden, 
                                map_s, next_map_s, pred_ego, next_ego, vision, next_vision]
                    
                    local_queue.append(line)
                    ego_queue.append(next_ego)
                    if self.pred_future_state:
  
                        fut_state_queue.append(next_obs)
                        fut_action_queue.append(action)
                        # if self.use_map:
                            # fut_map_queue.append(next_map_s)
                        if self.use_map:
                            fixed_next_map = self._adapt_map_state(next_map_s, self.neighbors, self.path_length, 5)
                            fut_map_queue.append(fixed_next_map)
                    else:
                        fut_state_queue=None
                        fut_action_queue=None
                        fut_map_queue=None


                    if frame_steps >= self.make_predictions:
                        assert len(list(local_queue))==self.make_predictions
                        assert len(list(ego_queue))==self.make_predictions
                        [obs, action, next_obs, reward, done_flag, mask, hidden, next_mask, next_hidden,
                            map_s, next_map_s, pred_ego, next_ego, vision, next_vision] = list(local_queue)[0]
        
                        #full egos trajs and full mask
                        if self.use_map:
                            egos = np.array(list(ego_queue))
                        else:
                            egos = None
                        # print(egos.shape)

                        if self.pred_future_state:
                            fut_action=np.array(list(fut_action_queue))
                            fut_state = np.array(list(fut_state_queue))
                            if self.use_map:
                                fut_map = np.array(list(fut_map_queue))
                            else:
                                fut_map = None
                        else:
                            fut_state=None
                            fut_map=None
                            fut_action=None

                        if self.use_map:
                            ego_mask = np.ones((self.make_predictions,))
                        else:
                            ego_mask = None

                        replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag,
                                        mask=mask, hidden=hidden, next_mask=next_mask, next_hidden=next_hidden,
                                        ego=egos,ego_mask=ego_mask,map_state=map_s,next_map_state=next_map_s,
                                        pred_ego=pred_ego, future_obs=fut_state, future_map_state=fut_map,
                                        future_action=fut_action, vision=vision, next_vision=next_vision)

                else:
                    ego_mask=None
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag,mask=mask,hidden=hidden,
                    next_mask=next_mask,next_hidden=next_hidden,ego=ego,ego_mask=ego_mask,map_state=map_s,next_map_state=next_map_s,
                    pred_ego=pred_ego,future_obs=None,future_map_state=None,future_action=None, vision=vision, next_vision=next_vision)

                obs = next_obs

                if self.bptt_hidden>0:
                    hidden = next_hidden
                if self.use_map:
                    map_s = next_map_s
                    ego = next_ego 

                vision = next_vision

            # end of a episode
            if done or episode_steps == self._episode_max_steps: 
                # if task is successful
                # success_log.append(1 if info[0] else 0)
                if isinstance(info, dict):
                    success_flag = info.get("finish", False)
                else:
                    success_flag = info[0]
                success_log.append(1 if success_flag else 0)

                #process the rest pairs and clear the queue:
                if self.make_predictions>0:
                    if self.use_map:
                        shape = list(ego_queue)[0].shape
                    if self.pred_future_state:
                        shape_a = list(fut_action_queue)[0].shape
                        shape_s = list(fut_state_queue)[0].shape
                        if self.use_map:
                            shape_m = list(fut_map_queue)[0].shape

                    for p in range(len(list(local_queue))):

                        [obs, action, next_obs, reward, done_flag, mask, hidden, next_mask, next_hidden,
                            map_s, next_map_s, pred_ego, next_ego, vision, next_vision] =local_queue.popleft()
                        #full egos trajs and full mask
                        
                        if self.pred_future_state:
                            if len(list(fut_state_queue))==0:
                                fut_state = np.zeros((self.make_predictions,)+shape_s)
                                fut_action = np.zeros((self.make_predictions,)+shape_a)
                                if self.use_map:
                                    fut_map = np.zeros((self.make_predictions,)+shape_m)
                            else:
                                fut_state = np.concatenate( (np.array(list(fut_state_queue)) , np.zeros((self.make_predictions - len(list(fut_state_queue)),)+shape_s) ),axis=0)
                                fut_action = np.concatenate( (np.array(list(fut_action_queue)) , np.zeros((self.make_predictions - len(list(fut_action_queue)),)+shape_a) ),axis=0)
                                if self.use_map:
                                    fut_map = np.concatenate( (np.array(list(fut_map_queue)) , np.zeros((self.make_predictions - len(list(fut_map_queue)),)+shape_m) ),axis=0)
                            
                            fut_state_queue.popleft()
                            fut_action_queue.popleft()
                            if self.use_map:
                                fut_map_queue.popleft()
                            assert len(list(local_queue))==len(list(fut_state_queue)),(len(list(local_queue)),len(list(fut_state_queue)))
                            assert len(list(local_queue))==len(list(fut_action_queue))
                            if self.use_map:
                                assert len(list(local_queue))==len(list(fut_map_queue))
                            else:
                                fut_map=None
                        else:
                            fut_state=None
                            fut_map=None
                            fut_action=None

                        if self.use_map:
                            if len(list(ego_queue))==0:
                                egos = np.zeros((self.make_predictions,)+shape)
                            else:
                                egos = np.concatenate( (np.array(list(ego_queue)) , np.zeros((self.make_predictions - len(list(ego_queue)),)+shape) ),axis=0)
                            # print(np.array(list(ego_queue)).shape)
                            ego_queue.popleft()
                            assert len(list(local_queue))==len(list(ego_queue))

                            ego_mask = [1]*len(list(ego_queue)) +[0]*(self.make_predictions - len(list(ego_queue)))
                        else:
                            egos,ego_mask = None,None

                        replay_buffer.add(
                                    obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag, mask=mask, 
                                    hidden=hidden, next_mask=next_mask, next_hidden=next_hidden, ego=egos, ego_mask=ego_mask, 
                                    map_state=map_s, next_map_state=next_map_s, pred_ego=pred_ego, future_obs=fut_state, 
                                    future_map_state=fut_map, future_action=fut_action, vision=vision, next_vision=next_vision)
                    
                    assert len(list(local_queue))==0
                    if self.use_map:
                        assert len(list(ego_queue))==0

                    if self.pred_future_state:
                        # fut_action_queue.clear()
                        
                        assert len(list(fut_state_queue))==0
                        assert len(list(fut_action_queue))==0
                        if self.use_map:
                            assert len(list(fut_map_queue))==0
                        
                replay_buffer.on_episode_end()
                obs = self._env.reset()
                reset_info = None
                if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[1], dict):
                    obs, reset_info = obs

                if self.use_map:
                    obs, ego, map_s = obs
                    map_s = self._adapt_map_state(map_s, neighbors=self.neighbors,
                                      path_len=self.path_length, target_dim=5)
                
                if isinstance(reset_info, dict):
                    vision = reset_info.get("vision", None)
                else:
                    vision = None
                next_vision = None

                if self.bptt_hidden>0:
                    hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
                else:
                    hidden,full_hidden,next_hidden=None,None,None
                
                # display info
                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                success = np.sum(success_log[-20:]) / 20

                self.return_log.append(episode_return)
                self.step_log.append(int(total_steps))
                self.train_success_rate.append(success)

                self.logger.info("Total Episode: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} Success: {4: 5.2f}, D:{5:5.2f} FPS:{6:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return,success,dis,fps))
                

                with self.tb_train.as_default():
                    tf.summary.scalar(name = 'return', data = episode_return, step = total_steps)
                    tf.summary.scalar(name = 'success', data = success, step = total_steps)
                    tf.summary.scalar(name = 'episode_len', data = episode_steps, step = total_steps) 
                
                # tf.summary.scalar(name="Common/training_return", data=episode_return)
                # tf.summary.scalar(name='Common/training_success', data=success)
                # tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

                # reset variables
                episode_returns.append(episode_return)
                dis = 0
                episode_steps = 0
                frame_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

                # save policy model
                if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
                    best_train = np.mean(episode_returns[-20:])

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)

                m = samples["mask"] if self.use_mask else None
                nm = samples["next_mask"] if self.use_mask else None
                h = samples["hidden"] if self.bptt_hidden>0 else None
                nh = samples["next_hidden"] if self.bptt_hidden>0 else None

                eg = samples['ego'] if self.make_predictions>0 and self.use_map else None
                ego_m = samples['ego_mask'] if self.make_predictions>0 and self.use_map else None

                mp = samples['map_state'] if self.use_map else None
                n_mp = samples['next_map_state'] if self.use_map else None
                pd = samples['pred_ego'] if self.multi_selection else None

                f_a = samples['future_action'] if self.pred_future_state else None
                f_s = samples['future_obs'] if self.pred_future_state else None
                f_m = samples['future_map_state'] if self.pred_future_state and self.use_map else None

                if self.pred_future_state and self.sep_train:
                    simi_loss = self._policy.train_rep(state=samples["obs"], map_state=mp,
                     future_state=f_s, future_map_state=f_m, future_action=f_a)
                    if total_steps % 500==0:
                        print(f'similarity_loss:{np.mean(simi_loss)}')
                
                with self.tb_train.as_default():
                    with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                        _,pred_traj,pred_loss = self._policy.train(
                            samples["obs"], samples["act"], samples["next_obs"],
                            samples["rew"], np.array(samples["done"], dtype=np.float32),
                            None if not self._use_prioritized_rb else samples["weights"],
                            mask=m,
                            hidden=h,
                            next_mask=nm,
                            next_hidden=nh,
                            ego=eg,
                            ego_mask=ego_m,
                            map_state=mp,
                            next_map_state=n_mp,
                            hist_traj=pd,
                            future_state=f_s, future_map_state=f_m, future_action=f_a
                            )
                        # if total_steps % 1000==0 and self.pred_future_state and not self.sep_train:
                        #     print(f'similarity_loss:{np.mean(pred_loss.numpy())/self.make_predictions}')
                    
                        # if total_steps%5000==0:
                        #     pass

                if self._use_prioritized_rb:
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        mask=m,
                        hidden=h,
                        next_mask=nm,
                        next_hidden=nh,
                        ego=eg,
                        ego_mask=ego_m,
                        map_state=mp,
                        next_map_state=n_mp)
                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)

            if total_steps % self._test_interval == 0:
                avg_test_return, avg_test_steps,success_rate = self.evaluate_policy(total_steps)
                self.eval_log.append(avg_test_return)
                self.test_step.append(avg_test_steps)
                self.test_success_rate.append(success_rate)

                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes,success_rate))    

                # tf.summary.scalar(name="Common/average_test_return", data=avg_test_return)
                # tf.summary.scalar(name="Common/average_test_episode_length", data=avg_test_steps)
                # tf.summary.scalar(name="Common/fps", data=fps)
                with self.tb_eval.as_default():
                    tf.summary.scalar("eval/avg_return",  avg_test_return,  step=total_steps)
                    tf.summary.scalar("eval/success_rate", success_rate,     step=total_steps)
                    tf.summary.scalar("eval/avg_steps",   avg_test_steps,    step=total_steps)

                # self.writer.flush()
                # garantir gravação em disco
                self.tb_train.flush()
                self.tb_eval.flush()
                
                # reset env
                obs = self._env.reset()

                if self.use_map:
                    obs, ego, map_s = obs
                    map_s = self._adapt_map_state(map_s, neighbors=self.neighbors,
                                      path_len=self.path_length, target_dim=5)

                hidden,full_hidden,next_hidden=None,None,None

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            # save checkpoint
            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy_continuously(self,plot_map_mode=False,map_dir=''):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)
        res = self.evaluate_policy(total_steps=0,plot_map_mode=plot_map_mode)
        print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                    res[0], res[1], self._test_episodes,res[2]))

    def evaluate_policy(self, total_steps,plot_map_mode=False):
        # tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())

        avg_test_return = 0.
        avg_test_steps = 0

        success_time = 0
        col_time = 0
        stag_time = 0
        ego_data=[]
        all_step=[]
        full_step =[]


        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size=self._episode_max_steps)

        for i in tqdm(range(self._test_episodes)):
            episode_return = 0.
            epi_step = 0
            epi_state=[]

            obs = self._test_env.reset()
            reset_info = None
            if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[1], dict):
                obs, reset_info = obs

            if self.use_map:
                print('================ ENTROU use_map =============')
                obs, ego, map_s = obs
                map_s = self._adapt_map_state(map_s, neighbors=self.neighbors,
                                  path_len=self.path_length, target_dim=5)

            hidden,full_hidden,next_hidden=None,None,None
            avg_test_steps += 1

            for j in range(self._episode_max_steps):
                if self.use_mask:
                    num = np.clip(j+1,0,self.timesteps)
                    mask = np.array([1]*num +[0]*(self.timesteps - num))
                    mask = np.expand_dims(mask, axis=0)
                else:
                    mask=None

                if epi_step%self.skip_timestep==0:
                
                    action = self._policy.get_action(np.expand_dims(obs,0),test=True,mask=mask,map_state=np.expand_dims(map_s,0))
                    act = action[0].numpy()
                
                next_obs, reward, done, info = self._test_env.step( act)
                line = obs[-1,:3]
                epi_state.append(line)

                if self.use_map:
                    next_obs,next_ego,next_map_s = next_obs
                    next_map_s = self._adapt_map_state(next_map_s, neighbors=self.neighbors,
                                       path_len=self.path_length, target_dim=5)

                avg_test_steps += 1
                epi_step += 1
                
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
                
                if self.bptt_hidden>0 :
                    next_hidden = np.expand_dims(full_hidden[(j)%self.timesteps],0)
                    if (j+1)%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)

                episode_return += reward
                obs = next_obs

                if self.use_map:
                    map_s = next_map_s
                if self.bptt_hidden>0:
                    hidden = next_hidden
                
                if done:
                    ego_data.append(epi_state)
                    event = info

                    if isinstance(event, dict):
                        if event.get("finish", False):
                            success_time += 1
                        if event.get("collision", False):
                            col_time += 1
                        if event.get("max_time", False):
                            stag_time += 1
                    else:
                        if event[0]:
                            success_time +=1
                            # all_step.append(epi_step)
                        if event[1]:
                            col_time +=1
                        if event[-1]:
                            stag_time +=1
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            avg_test_return += episode_return
        
        s_r,c_r,stag = success_time/self._test_episodes , col_time/self._test_episodes , stag_time/self._test_episodes
        
        print(f'mean_return:{avg_test_return / self._test_episodes}'
              f'success rate:{s_r}'
              f'collision rate:{c_r}'
              f'stagnation:{stag}'
              )
        
        step_for_tb = total_steps
        if step_for_tb == 0:
            # usar o passo do ckpt se esta avaliação veio de um checkpoint específico
            step_for_tb = self._ckpt_step_from_path(getattr(self, "_latest_path_ckpt", None) or self._model_dir) or 0

        with self.tb_eval.as_default():
            tf.summary.scalar("eval/avg_return",  avg_test_return / self._test_episodes, step=step_for_tb)
            tf.summary.scalar("eval/success_rate", success_time / self._test_episodes,    step=step_for_tb)
            tf.summary.scalar("eval/avg_steps",   avg_test_steps / self._test_episodes,   step=step_for_tb)
        self.tb_eval.flush()

        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes,success_time / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps

        if args.episode_max_steps is not None:
            self._episode_max_steps = args.episode_max_steps
        else:
            self._episode_max_steps = args.max_steps
            
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(5e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        # parser.add_argument('--normalize-obs', action='store_true', default=False,
                            # help='Normalize observation')
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
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        parser.add_argument('--scenario',default='')
        
        return parser