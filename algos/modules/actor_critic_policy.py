
import tensorflow_probability as tfp
from .policy import RLEncoder
# from tf2rl.misc.target_update_ops import update_target_variables
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

class SAC_Critic(tf.keras.Model):
    def __init__(
        self,
        params, 
        state_shape, 
        action_dim, 
        hidden_activation="relu", 
        name='actor_critic_network'):

        super().__init__(name=name)
        head_num=params['head_num'] 
        self.encoder = RLEncoder(state_shape,
                                 units=params['units'],
                                 state_input=params['state_input'],
                                 lstm=params['LSTM'],
                                 trans=False,
                                 cnn_lstm=params['cnn_lstm'],
                                 bptt=params['bptt'],
                                 ego_surr=params['ego_surr'],                    
                                 neighbours=params['neighbours'],
                                 time_step=params['time_step'],
                                 debug=False,
                                 make_rotation=params['make_rotation'],
                                 use_map=params['use_map'],
                                 num_traj=params['num_traj'],
                                 path_length=params['path_length'],
                                 num_heads=head_num,
                                 cnn=params['cnn'],
                                 use_hier=params['use_hier'],
                                 random_aug=params['random_aug'],
                                 carla=params['carla'],
                                 no_ego_fut=params['no_ego_fut'],
                                 no_neighbor_fut=params['no_neighbor_fut'], 
                                 use_vision=params.get('use_vision', False), 
                                 vision_dim=params.get('vision_dim', 280))
        
        self.params = params

        self.multi_selection = 'multi_selection' in list(params.keys())
        
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        self.use_map= params['use_map'] or params['use_hier']
  
        self.critic_layers = [layers.Dense(128, activation=hidden_activation), 
                                layers.Dense(32, activation=hidden_activation),
                                layers.Dense(1)]
        
        self.action_layer = [layers.Dense(64, activation=hidden_activation)]

        self.critic_layers_2 = [layers.Dense(128, activation=hidden_activation), 
                                layers.Dense(32, activation=hidden_activation),
                                layers.Dense(1)]
        
        self.action_layer_2 = [layers.Dense(64, activation=hidden_activation)]

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map'] or params['use_hier']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state, dummy_action, mask=m, init_state=init_state, map_state=map_s)
        h = self._get_hidden(dummy_state, mask=m, init_state=init_state, map_state=map_s)
        self.h_size = tuple(h.get_shape().as_list())
        self.summary()

    def call(
            self,
             states,
             actions,
             mask=None,
             test=False,
             init_state=None,
             map_state=None,
             traj=None,
             curr_frames=None,
             en_input=False,
             aug=True, 
             vision=None):

        feat, _ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state,aug=aug, vision=vision)

        if self.use_map:
            ensembles = feat.get_shape()[1]
            act_feature = tf.expand_dims(self.action_layer[0](actions),1)
            features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=-1)
            act_feature_2 = tf.expand_dims(self.action_layer_2[0](actions),1)
            features_2 =  tf.concat([feat, tf.repeat(act_feature_2,ensembles,axis=1)], axis=-1)
            if self.multi_selection:
                raise NotImplementedError('multi selection have not been implemented!')
        else:
            act_feature = self.action_layer[0](actions)
            features =  tf.concat([feat, act_feature], axis=1)

            act_feature_2 = self.action_layer_2[0](actions)
            features_2 =  tf.concat([feat, act_feature_2], axis=1)

        for layer in self.critic_layers:
            features = layer(features)

        for layer in self.critic_layers_2:
            features_2 = layer(features_2)
        if self.use_map:
            features = tf.reduce_mean(features,axis=1)
            features_2 = tf.reduce_mean(features_2,axis=1)

        values = tf.squeeze(features, axis=-1)
        values_2 = tf.squeeze(features_2, axis=-1)
        return values,values_2
    
    def _get_hidden(
            self,
            states,
            mask=None, 
            test=False,
            init_state=None,
            map_state=None,
            aug=True, 
            vision=None):
        features,_ = self.encoder(
                            states,
                            mask=mask, 
                            test=test,
                            init_state=init_state,
                            map_state=map_state,
                            aug=aug,
                            vision=vision)
        return features #[:,:,128:]
    
    
    def _get_hidden_and_val(
                            self,
                            states,
                            mask=None, 
                            test=True,
                            init_state=None,
                            map_state=None,
                            aug=True,
                            vision=None):
        
        features,val = self.encoder(
                                states,
                                mask=mask, 
                                test=test,
                                init_state=init_state,
                                map_state=map_state,
                                aug=aug,
                                vision=vision)
        return features, val
    

    def _get_pca_val(
                    self,
                    states,
                    actions,
                    mask=None, 
                    test=True,
                    init_state=None,
                    map_state=None,
                    aug=True,
                    vision=None):
        
        feat,_ = self.encoder(
                            states,
                            mask=mask, 
                            test=test,
                            init_state=init_state,
                            map_state=map_state,
                            aug=aug,
                            vision=vision)
        
        ensembles = feat.get_shape()[1]
        act_feature = tf.expand_dims(self.action_layer[0](actions),1)
        features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=-1)

        act_feature_2 = tf.expand_dims(self.action_layer_2[0](actions),1)
        features_2 =  tf.concat([feat, tf.repeat(act_feature_2,ensembles,axis=1)], axis=-1)

        out_feat = 0.5*(features + features_2)
        out_feat=tf.squeeze(out_feat,axis=1)

        for layer in self.critic_layers:
            features = layer(features)

        for layer in self.critic_layers_2:
            features_2 = layer(features_2)
        
        values = tf.squeeze(features, axis=-1)
        values_2 = tf.squeeze(features_2, axis=-1)

        return out_feat, 0.5*(values+values_2)

class SAC_Actor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(
                self,
                params,
                state_shape,
                action_dim, 
                max_action,
                hidden_activation="relu", 
                name='gaussian_policy',
                state_independent_std=False,
                squash=True,
                drop_rate=0.1,
                ensembles=False):
        
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self._squash = squash
        self.use_map=params['use_map'] or params['use_hier']
        self.params = params
        self.ensemble = ensembles

        self.multi_selection = 'multi_selection' in list(params.keys())
        self._max_action = max_action

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")
        self.actor_layers = [layers.Dense(128, activation=hidden_activation),
                             layers.Dense(32, activation=hidden_activation)]
        
        dummy_state = tf.constant(np.zeros(shape=state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))
        
        if params['use_map']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state)
        self.summary()
    
    def call(self, state, test=False):
        features = tf.stop_gradient(state)

        for layer in self.actor_layers:
            features = layer(features)
        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)
            std = tf.exp(log_std)

        if self.use_map:
            mean = tf.reduce_mean(mean, axis=1)
            std = tf.reduce_mean(std, axis=1)
                
        dist =  tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

        if test:              
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)
        entropy = dist.entropy()

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=-1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action

        return actions, log_pis, entropy
    


class Represent_Learner(tf.keras.Model):
    def __init__(
                self,
                params,
                state_shape,
                action_dim,
                qf,
                qf_target,
                dim_shape,
                hidden_activation="relu", 
                name='Represent_Learner',
                use_sep_tar_encoder=False,
                use_q_head=False):
        
        super().__init__(name=name)
        self.params = params
        self.recurrent_layer = layers.MultiHeadAttention(num_heads=params['head_num'], 
                                    key_dim=128//params['head_num'], output_shape=dim_shape)
        
        self.action_layer = layers.Dense(64, activation=hidden_activation, name='action_0')

        # self.transition_layers = [layers.Dense(256, activation=hidden_activation),
        #                             layers.Dense(dim_shape, activation=hidden_activation)]

        if use_q_head:
            self.projection_layers=[qf.critic_layers[0]]
            self.projection_layers_target=[qf_target.critic_layers[0]]
        else:
            self.projection_layers = [layers.Dense(256, activation=hidden_activation, name='proj_1'),
                                        layers.Dense(dim_shape, activation=hidden_activation, name='proj_2')]
            
            if not params['random_aug']:
                self.projection_layers_target=[layers.Dense(256, activation=hidden_activation,name='t_proj_1'),
                                                layers.Dense(dim_shape, activation=hidden_activation,name='t_proj_2')]
                
        self.use_q_head = use_q_head

        if not (use_q_head or params['random_aug']):
            update_target_variables(self.projection_layers_target.weights,
                                    self.projection_layers.weights, tau=1.0)

        self.encoder = qf.encoder

        if use_sep_tar_encoder:
            if params['make_prediction']:
                head_num=1 #params['head_num']
            else:
                head_num=params['head_num']

            self.target_encoder = RLEncoder(state_shape, units=[256]*3, state_input=params['state_input'],
                                            lstm=params['LSTM'], trans=False, cnn_lstm=params['cnn_lstm'],
                                            bptt=params['bptt'],ego_surr=params['ego_surr'], neighbours=params['neighbours'],
                                            time_step=params['time_step'], debug=False, make_rotation=params['make_rotation'],
                                            use_map=params['use_map'],num_traj=params['num_traj'],path_length=params['path_length'],
                                            num_heads=head_num,cnn=params['cnn'], use_hier=params['use_hier'],
                                            random_aug=params['random_aug'], carla=params['carla'],)
        else:
            self.target_encoder = qf_target.encoder

        self.pred_layer = layers.Dense(128,name='prediction_head')

        self.pred_step = params['future_step']

        self.similarity_loss = tf.keras.losses.cosine_similarity

        dummy_state = tf.constant(np.zeros(shape=(1,self.pred_step,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,self.pred_step,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,self.pred_step,dummy_state.get_shape()[1]])

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map'] or params['use_hier']:
            map_s = tf.constant(np.zeros(shape=(1,self.pred_step,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
            self(dummy_state[:,0],map_s[:,0],dummy_action,dummy_state,map_s)
        else:
            map_s = None
            self(dummy_state[:,0],None,dummy_action,dummy_state,map_s)
        
        self.summary()
    
    def _timestep_attention(self,states,training,mask):
        mask = tf.cast(mask, tf.int16)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        length = mask.get_shape()[-1]
        a = tf.cast(tf.expand_dims(tf.linalg.band_part(tf.ones((length,length)),-1,0),axis=0),tf.int16)
        mask = tf.multiply(a,mask)
        state_val = self.recurrent_layer(states,states,attention_mask=mask,training=True)
        return state_val
        
    def _action_transition(self, feat, actions, init_state=None, re=False):

        if self.params['use_map'] or self.params['use_hier']:
            ensembles = feat.get_shape()[1]
            act_feature = tf.expand_dims(self.action_layer(actions),1)
            features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=2)
        else:
            act_feature = self.action_layer(actions)
            features =  tf.concat([feat, act_feature], axis=-1)

        for layer in self.transition_layers:
            features = layer(features)
        return features
    
    def _make_recurrent_rep(
                            self,
                            states,
                            map_state,
                            actions,
                            next_states,
                            next_map_state,
                            mask = None, 
                            test = False,
                            init_state = None,
                            vision = None,
                            next_vision = None):

        f_z = []
        f_z_f = []

        for i in range(self.pred_step):

            if i==0:
                f,_ = self.encoder(states,
                                   mask=mask, 
                                   test=test,
                                   init_state=init_state,
                                   map_state=map_state,
                                   vision=vision)
            else:
                f,_ = self.encoder(next_states[:,i-1],
                                   mask=mask, 
                                   test=test,
                                   init_state=init_state,
                                   map_state=next_map_state[:,i],
                                   vision=next_vision[:,i-1] if next_vision is not None else None)
            f_z.append(f)

            if self.params['random_aug']:
                f_n,_ = self.encoder(next_states[:,i],
                                     mask = mask, 
                                     test = test,
                                     init_state = init_state,
                                     map_state = next_map_state[:, i-1],
                                     curr_frames = None,
                                     vision = next_vision[:, i-1] if next_vision is not None else None)
            else:
                f_n,_ = self.target_encoder(next_states[:,i],
                                            mask=mask, 
                                            test=test,
                                            init_state=init_state,
                                            map_state=next_map_state[:,i],
                                            curr_frames=None,
                                            vision=next_vision[:,i] if next_vision is not None else None)
            f_z_f.append(f_n)

        f_z,f_z_f = tf.stack(f_z,1) , tf.stop_gradient(tf.stack(f_z_f,1))
        step_mask = tf.cast(tf.not_equal(next_states[:,:,0,0,0], 0),tf.float32)
        f_z = f_z[:,:,0,:]
        f_z_f = f_z_f[:,:,0,:]
        f_a = self.action_layer(actions)
        p_f_z = tf.concat([f_z,f_a], axis=-1)
        z = self._timestep_attention(p_f_z,training=True,mask=step_mask)
        p_f_z,f_z_f = self._projection(z,f_z_f)
        p_f_z,f_z_f = self.pred_layer(p_f_z),tf.stop_gradient(f_z_f)
        step_mask = tf.cast(step_mask,tf.float32)
        loss = tf.reduce_mean(tf.multiply(step_mask,self.similarity_loss(f_z_f,p_f_z,axis=-1)))
        return loss
    

    def _projection(self,feat,feat_target):

        if self.params['random_aug']:
            for layer in self.projection_layers:
                feat,feat_target = layer(feat), layer(feat_target)
        else:
            for layer, target_layer in zip(self.projection_layers,self.projection_layers_target):
                feat,feat_target = layer(feat), target_layer(feat_target)
        return feat,feat_target
    
        
    def call(self,
             states,
             map_state,
             actions,
             next_states,
             next_map_state,
             mask = None, 
             test = False,
             init_state = None,
             vision = None,
             next_vision = None):
        
        return self._make_recurrent_rep(states, map_state, actions, next_states, next_map_state, 
                                        mask, test, init_state, vision, next_vision)
    
    def _update_params(self, tau):

        if not self.params['random_aug']:
            return update_target_variables(self.projection_layers_target.weights,
                                            self.projection_layers.weights, tau)