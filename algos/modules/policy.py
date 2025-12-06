import numpy as np
import tensorflow as tf
layers = tf.keras.layers

class Rotater(layers.Layer):    # Data Augmentation by Rotation
    def __init__(
        self, 
        name='rotater', 
        mode='traj', 
        aug=True, 
        carla=True):
        super().__init__(name=name)
        self.mode = mode
        self.random_aug = aug
        self.carla = carla

    def call(self, states, mask, curr_frame=None, aug=True, random_angle=None):
        if self.mode=='traj':
            return self._make_rotations(states, mask,curr_frame,aug)
        elif self.mode=='fut':
            return self._rotate_fut(states,curr_frame,aug)
        else:
            return self._make_map_rotations(states, mask, curr_frame,aug,random_angle)

    def _rotate_fut(self, states, curr_frames, aug=True):
        '''
        states: [batch, timestep, 2]
        curr_frames: [batch, 5]
        '''
        mask = tf.not_equal(states, 0)[:, :,0]
        mask = tf.cast(mask,tf.float32)

        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1])
        #[batch,6,time_steps]
        x = states[:,:,0] - tf.reshape(curr_frames[:,0], [-1,1])
        y = states[:,:,1] - tf.reshape(curr_frames[:,1], [-1,1])

        new_x =   tf.multiply(cos_a,x) + tf.multiply(sin_a,y)
        new_y = - tf.multiply(sin_a,x) + tf.multiply(cos_a,y)

        rotated_state = [tf.expand_dims(new_x, 2), tf.expand_dims(new_y, 2)]
        rotated_state =  tf.concat(rotated_state, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

        return tf.multiply(mask, rotated_state) , mask

        
    def _make_rotations(self, states, mask, curr_frame=None, aug=True):
        """
        ref: lhc_mtp_loss.py 
        make clock-wise rotation after moving to the ego point
        states: [batch, ego + neighbours, timesteps, state_dim]
        mask: [batch, ego + neighbors, timestep]
        """

        # get curr_frame location according to the mask
        mask = tf.cast(mask, tf.int32)
        ind = tf.reduce_sum(mask[: ,0, :], axis=-1)
        ind = tf.clip_by_value(ind - 1, 0, 100)

        #gather indices is [batch_no , 0(ego) , mask_ind]
        #return [batch,hidden(4)]
        if curr_frame is None:
            curr_frames = tf.gather_nd(states, tf.transpose([tf.range(mask.get_shape()[0]), tf.zeros_like(ind), ind,]))
        else:
            #for future representations
            curr_frames = curr_frame
        if self.random_aug:
            r_range = np.pi/2
            random_angle =  tf.random.uniform(shape=curr_frames[:,2].get_shape(),minval=-r_range,maxval=r_range)
            cos_r = tf.reshape(tf.math.cos(random_angle),[-1,1,1])
            sin_r = tf.reshape(tf.math.sin(random_angle),[-1,1,1])
        

        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1,1])

        #[batch, 6, time_steps]
        
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1]) 
        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])
        angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])

        if aug:
            new_x =   tf.multiply(cos_a,x) + tf.multiply(sin_a,y)
            new_y = - tf.multiply(sin_a,x) + tf.multiply(cos_a,y)
            if self.random_aug:
                n_x =   tf.multiply(cos_r,new_x) + tf.multiply(sin_r,new_y)
                n_y = - tf.multiply(sin_r,new_x) + tf.multiply(cos_r,new_y)
                new_x,new_y = n_x,n_y
        else:
            new_x,new_y = x,y

        vx = states[:,:,:,3] - tf.reshape(curr_frames[:,3],[-1,1,1])
        vy = states[:,:,:,4] - tf.reshape(curr_frames[:,4],[-1,1,1])
        if aug:
            new_vx =   tf.multiply(cos_a,vx) + tf.multiply(sin_a,vy)
            new_vy = - tf.multiply(sin_a,vx) + tf.multiply(cos_a,vy)
            if self.random_aug:
                n_vx =   tf.multiply(cos_r,new_vx) + tf.multiply(sin_r,new_vy)
                n_vy = - tf.multiply(sin_r,new_vx) + tf.multiply(cos_r,new_vy)
                new_vx,new_vy = n_vx,n_vy
        else:
            new_vx,new_vy = vx,vy

        rotated_state = [
            tf.expand_dims(-new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(angle,3),
            tf.expand_dims(-new_vx,3),
            tf.expand_dims(new_vy,3)
        ]
        if not self.carla:
            rotated_state = [
            tf.expand_dims(new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(angle,3),
            tf.expand_dims(new_vx,3),
            tf.expand_dims(new_vy,3)
        ]

        mask = tf.cast(tf.expand_dims(mask, axis=-1),tf.float32)
        rotated_state =  tf.concat(rotated_state, axis=-1)
        
        if self.random_aug:
            return tf.multiply(mask, rotated_state),curr_frames,random_angle
        else:
            return tf.multiply(mask, rotated_state),curr_frames,None
    
    def _make_map_rotations(self, states, mask, curr_frames, aug=True, random_angle=None):  # curr_frames: [batch, 5]
        mask = tf.cast(mask, tf.int32)   # [batch, polyline, timestep]
        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw), [-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw), [-1,1,1])
        # angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1])
        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])

        if self.random_aug: # True
            cos_r = tf.reshape(tf.math.cos(random_angle),[-1,1,1]) 
            sin_r = tf.reshape(tf.math.sin(random_angle),[-1,1,1])
        

        if aug: # True # rotate according to ego yaw
            new_x =tf.multiply(cos_a,x) + tf.multiply(sin_a,y)  # (batch,polyline,timestep)
            new_y =-tf.multiply(sin_a,x) + tf.multiply(cos_a,y) # (batch,polyline,timestep)
            if self.random_aug:
                n_x =   tf.multiply(cos_r,new_x) + tf.multiply(sin_r,new_y)
                n_y = - tf.multiply(sin_r,new_x) + tf.multiply(cos_r,new_y)
                new_x,new_y = n_x,n_y
        else:
            new_x,new_y = x,y
    
        rotated_state = [tf.expand_dims(new_x, -1),tf.expand_dims(new_y, -1),]
        if not self.carla:
            angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])
            rotated_state = [
                tf.expand_dims(new_x, 3),
                tf.expand_dims(new_y, 3),
                tf.expand_dims(angle, 3),
                tf.expand_dims(states[:,:,:,3], 3),
                tf.expand_dims(states[:,:,:,4], 3)
            ]
        mask = tf.cast(tf.expand_dims(mask, axis=-1),tf.float32) # [batch, polyline, timestep, 1]
        rotated_state =  tf.concat(rotated_state, axis=-1) # [batch, polyline, timestep, dim]


        return tf.multiply(mask, rotated_state)  # [batch, polyline, timestep, dim]
    

class MapEncoder(layers.Layer):
    def __init__(
        self, 
        return_attention_scores=True, 
        drop_rate=0, 
        carla=True):
        super().__init__()
        self.return_attention_scores = return_attention_scores
        self.node_attention = layers.MultiHeadAttention(num_heads=2, key_dim=128, dropout=drop_rate, output_shape=64*3) 
        self.flatten = layers.GlobalMaxPooling1D()
        self.vector_feature = layers.Dense(units=64, activation='relu')
        self.sublayer = layers.Dense(units=128, activation='relu')
        self.carla = carla

    def call(self, inputs, mask, test):   # inputs: [batch, num_nodes, dim] (32, 10, 2)
        mask = tf.cast(mask, tf.int16)    # (batch, num_nodes) # (32, 10)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :]) # (batch, num_nodes, num_nodes) # (32, 10, 10)

        if self.carla:
            nodes = inputs[:, :, :2]   # (batch, num_nodes, 2) # (32, 10, 2)
        else:
            nodes = inputs[:, :, :3]


        if self.return_attention_scores:
            attention_output, attention_scores = self.node_attention(query=nodes, value=nodes, 
                                                    attention_mask=mask, training=bool(1-test),
                                                    return_attention_scores=self.return_attention_scores)
            # attention_output: (batch, num_nodes, dim) 
            # attention_scores: (batch, num_heads, num_nodes, num_nodes) (32, 2, 10, 10)
        else:
            attention_output = self.node_attention(query=nodes, value=nodes, attention_mask=mask,
                                        training=bool(1-test), return_attention_scores=self.return_attention_scores)
            
        nodes = tf.nn.relu(attention_output)   # (batch, num_nodes, dim) # (32, 10, 192)
        nodes = self.flatten(nodes)            # (batch, dim) # (32, 192)
        vector = self.vector_feature(inputs[:, 0, -2:])  # (batch, 64)
        out = tf.concat([nodes, vector], axis=1) # (batch, 192 + 64)
        polyline_feature = self.sublayer(out) # (batch, 256)

        if self.return_attention_scores:
            # print(val.get_shape())
            attention_scores = tf.reduce_mean(tf.reduce_mean(attention_scores, axis=1), axis=1)
            return polyline_feature, attention_scores # polyline_feature: (batch, 128), attention_scores: (batch, num_nodes, num_nodes)
        else:
            return polyline_feature 
        
        
class VisionEncoder(tf.keras.Model):
    def __init__(self, out_dim=128, name="vision_encoder"):
        super().__init__(name=name)
        self.net = tf.keras.Sequential([layers.Dense(256, activation='relu'),
                            layers.Dense(out_dim, activation='relu')])

    def call(self, vision_vec):
        return self.net(vision_vec)  # (batch, out_dim) 
    

class Hierachial_Transformer(tf.keras.Model):
    def __init__(
        self, 
        state_shape, 
        name='hier_encoder', 
        units=256, 
        num_heads=2, 
        drop_rate=0, 
        neighbours=5, 
        make_rotation=True, 
        time_step=10, 
        num_traj=1, 
        random_aug=True, 
        no_ego_fut=False, 
        no_neighbor_fut=False, 
        carla=True):    # state_shape = (6, 10, 5)

        super().__init__(name=name)   
        self.map_layer = MapEncoder(return_attention_scores=True, drop_rate=drop_rate, carla=carla)
        self.neighbours = neighbours
        self.make_rotation = make_rotation
        self.time_step = time_step
        self.time_layer = layers.MultiHeadAttention(num_heads, units, dropout=drop_rate, output_shape=units)
        self.time_pooling = layers.GlobalMaxPooling1D()
        self.rel_layer = layers.MultiHeadAttention(num_heads, units, dropout=drop_rate, output_shape=units)
        self.map_attention = layers.MultiHeadAttention(num_heads, units, dropout=drop_rate, output_shape=units)
        self.num_traj = num_traj
        self.no_ego_fut = no_ego_fut
        self.no_neighbor_fur = no_neighbor_fut
        self.carla = carla

        if self.make_rotation:
            self.rotater = Rotater(mode='traj', aug=random_aug, carla=carla)
            self.map_rotater = Rotater(mode='map', aug=random_aug, carla=carla)

        self.final_attention = []
        for _ in range(self.num_traj):
            self.final_attention.append(layers.MultiHeadAttention(num_heads, units, dropout=drop_rate, output_shape=units)) # len(self.final_attention) = num_traj

        if self.carla:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*3, time_step, 2), dtype=np.float32))  # shape=(32, 18, 10, 2)
        else:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*2, time_step, 5), dtype=np.float32)) 


        dummy_state = tf.constant(np.zeros(shape=(32,) + state_shape, dtype=np.float32))  # (32, 6, 10, 5)
        self(dummy_state, map_state=map_s)  # Initialize weights the layers
        self.summary()   # present archicteture 
        

    def call(self,states, test=False, map_state=None, aug=True): 

        training= bool(1 - test) 
        mask = tf.not_equal(states, 0)[:, :, :, 0]   # mask: [batch, ego+neighbors, time_step]
        if self.make_rotation:
            states, curr_frames, rg = self.rotater(states, mask, aug=aug)

        ego_states , neighbor_states = states[:, 0, :, :] , states[:, 1:, :, :]  # ego_states: [batch, time_step, state_dim], neighbor_states: [batch, neighbors, time_step, state_dim]
        ego_mask, neighbor_mask = mask[:, 0, :],  mask[:, 1:, :]  # ego_mask: [batch, time_step], neighbor_mask: [batch, neighbors, time_step]

        actor_mask = tf.not_equal(tf.concat([tf.expand_dims(tf.ones_like(ego_states), 1), neighbor_states], axis=1), 0)[:, :, 0, 0]  # (batch, ego+neighbors)

        ego = self._timestep_attention(ego_states, training, ego_mask)  # ego: [batch, units]

        neighbors = []
        for i in range(self.neighbours):
            neighbors.append(self._timestep_attention(neighbor_states[:,i,:,:], training, neighbor_mask[:,i,:])) # neighbors: list of [batch, units] 

        map_mask = tf.not_equal(map_state, 0)[:,:,:,0] # (32, 12, 10) 
        map_traj_mask = tf.not_equal(map_state, 0)[:,:,0,0]  # (32, 12)
        
        if self.make_rotation:
            map_state = self.map_rotater(map_state,map_mask,curr_frames,aug,random_angle=rg) # self._make_map_rotations(states, mask, curr_frame,aug,random_angle)

        map = []
        for i in range(map_state.get_shape().as_list()[1]): # map_state.get_shape().as_list()[1] = 12
            map_layer = self.map_layer(map_state[:, i], map_mask[:, i, :],test)[0] # (32, 256)
            map.append(map_layer) 

        if test:
            val = []
            for i in range(map_state.get_shape().as_list()[1]): # map_state.get_shape().as_list()[1] = 12
                v = self.map_layer(map_state[:, i], map_mask[:, i, :],test)[1]
                # map_state[:, i].shape = (32, 10, 2) , map_mask[:, i, :].shape = (32, 10) 
                val.append(v) # len(val) = 12
            # val = [self.map_layer(map_state[:, i], map_mask[:, i, :],test)[1] for i in range(map_state.get_shape().as_list()[1])]
            val = tf.stack(val,axis=1)
        #(b,12,256)
        map = tf.stack(map, axis=1) # (32, 12, 256)
        
        if self.carla:
            ego_map, neighbor_map = map[:,:3,:], map[:,3:,:]
            ego_map_traj_mask, neighbor_map_traj_mask = map_traj_mask[:,:3], map_traj_mask[:,3:]
        else:
            ego_map,neighbor_map = map[:,:2,:],map[:,2:,:]
            ego_map_traj_mask,neighbor_map_traj_mask = map_traj_mask[:,:2],map_traj_mask[:,2:]
        # ego_map_mask,neighbor_map_mask = map_mask[:,:2,:],map_mask[:,2:,:]

        neighbor_rel_val = []
        for i in range(self.neighbours):
            mv_rel, mv_val = self._map_vehicle_rel(neighbors[i],neighbor_map,neighbor_map_traj_mask,i*2)
            # neighbors[i]: (batch, units), neighbor_map: (batch, num_polylines, dim), neighbor_map_traj_mask: (batch, num_polylines)
            # mv_rel: (batch, units), mv_val: (batch, 2)
            neighbor_rel_val.append(mv_rel)  # len(neighbor_rel_val) = 5

        if self.no_neighbor_fur:
            neighbor_rel_val = neighbors


        neighbor_val = []
        for i in range(self.neighbours):
            mv_rel, mv_val = self._map_vehicle_rel(neighbors[i], neighbor_map, neighbor_map_traj_mask, i*2)
            neighbor_val.append(mv_val)
            

        actor = tf.concat([ego[:, tf.newaxis], tf.stack(neighbor_rel_val, axis=1)], axis=1) # (batch, ego+neighbors, units) (32, 6, 256)
        actor_rel = self.rel_layer(tf.expand_dims(ego, axis=1), actor, attention_mask=actor_mask[:, tf.newaxis], training=training) # cross-attention entre o ego e os atores vizinhos
        # tf.expand_dims(ego, axis=1) # (batch, 1, units) (32, 1, 256)
        # actor_rel: (batch, 1, units) (32, 1, 256)
        actor_rel = tf.nn.relu(tf.squeeze(actor_rel,axis=1)) # (batch, units) (32, 256)

        goals,ego_val = self._goal_layer(actor_rel[:,tf.newaxis], ego_map, ego_map_traj_mask[:,tf.newaxis]) # cross-attention entre o ego e os mapas
        # ego_states = tf.concat([actor_rel, ego], axis=-1)
        # goals (batch, num_traj, units) (32, 1, 256)
        ego_states = tf.repeat(actor_rel[:, tf.newaxis], self.num_traj, axis=1) # ego_states: (batch, num_traj, units) (32, 1, 256)

        if self.no_ego_fut:
            states = ego_states
        else:
            states = goals + ego_states #+ ego
        if test:
            neighbor_val = [ego_val] + neighbor_val
            neighbor_val = tf.expand_dims(tf.concat(neighbor_val,axis=-1),axis=-1)
            return states,neighbor_val #tf.multiply(neighbor_val,val)

        return states

    def _timestep_attention(self, states, training, mask):  # states.shape = [batch, timesteps, dim]
        mask = tf.cast(mask, tf.int16) # (batch,timesteps)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :]) # (batch,timesteps,timesteps) # mascara de atenção bidirecional (importante!!!!)
        state_val = self.time_layer(states, states, attention_mask=mask, training=training)  # self.time_layer é multihead attention (self-attention temporal) # (batch, timesteps, units)
        state_val = tf.nn.relu(state_val) # (batch, timesteps, units)
        state_val = self.time_pooling(state_val) # (batch, units) # resumo temporal por max pooling
        return state_val # (batch, units)
    
    def _map_vehicle_rel(self,value,map_state,map_mask,i):
        use_map = map_state[:,i:i+2,:] # (batch, 2, dim) (32, 2, 256)
        use_map_mask = tf.concat([tf.ones_like(map_mask[:,0])[:,tf.newaxis],map_mask[:,i:i+2]],axis=1)[:,tf.newaxis] # (batch, 1, 3) (32, 1, 3)
        mv_rel = tf.concat([value[:, tf.newaxis], use_map], axis=1)  # value[:, tf.newaxis] (batch, 1, units)  # value/key: veículo  # use_map: 2 mapas mais próximos
        # mv_rel: veículo + 2 mapas mais próximos # (batch, 3, dim)
        mv_val,val = self.map_attention(value[:,tf.newaxis],mv_rel,attention_mask=use_map_mask,training=True,return_attention_scores=True) # cross-attention entre o veículo e os mapas
        # mv_val: (batch, 1, units) (32, 1, 256), val: (batch, num_heads, 1, 3) (32, 2, 1, 3)
        val = tf.reduce_mean(tf.squeeze(val,axis=-2),axis=1)[:,1:] # atenção média sobre os mapas (ignorando o veículo)  # val: (batch, 2) (32, 2)
        # print(val.get_shape())
        mv_val = tf.squeeze(mv_val,axis=1) # (batch, units) (32, 256)
        mv_val = tf.nn.relu(mv_val) # (batch, units)
        return mv_val,val
    
    def _goal_layer(self, query, key, mask=None, training=True):
        output, v = [], []
        for i in range(self.num_traj): # self.num_traj = num_traj = 1
            value, val = self.final_attention[i](query, key, attention_mask=mask, return_attention_scores=True, training=training)
            # value: (batch, 1, units) , val: (batch, num_heads, 1, num_polylines)
            output.append(tf.squeeze(value, axis=1))  # (batch, units)
            v.append(val)
        v = tf.reduce_mean(tf.squeeze(v[0], axis=-2), axis=1)
        # tf.squeeze(v[0], axis=-2): (batch, num_heads, num_polylines)
        # v: (batch, num_polylines)
        value = tf.nn.relu(tf.stack(output, axis=1))
        # tf.stack(output, axis=1): (batch, num_traj, units)Para_MTPLoss
        # value: (batch, num_traj, units) (32, 1, 256)
        return value, v
    

class RLEncoder(tf.keras.Model):
    def __init__(
            self, 
            state_shape, 
            # units=[256]*3, 
            units = 128,
            name='rl_encoder', 
            state_input=False, 
            lstm=False, 
            trans=False,
            cnn_lstm=False, 
            bptt=False, 
            ego_surr=False, 
            neighbours=5, 
            time_step=10, 
            debug=False,
            make_rotation=True, 
            use_mask=False, 
            use_map=True, 
            num_traj=1, 
            cnn=False, 
            path_length=0, 
            num_heads=2, 
            use_hier=False, 
            random_aug=False, 
            no_ego_fut=False, 
            no_neighbor_fut=False, 
            carla=True,
            use_vision=True,
            vision_dim=128):
        
        super().__init__(name=name)

        self.lstm = lstm  
        self.cnn = cnn  
        self.cnn_lstm = cnn_lstm
        self.state_input = state_input
        self.bptt=bptt
        self.ego_surr=ego_surr        
        self.trans = trans
        self.debug=debug
        self.use_map=use_map
        self.neighbours = neighbours
        self.num_traj = num_traj
        self.use_mask=use_mask
        self.use_hier = use_hier
        self.carla = carla

        self.rep_dim = 128

        self.use_vision = use_vision
        self.vision_dim = vision_dim

        if self.use_vision:
            self.vision_encoder = VisionEncoder(out_dim=self.rep_dim)
            self.vision_fusion = layers.Dense(self.rep_dim, activation='relu')

   
        if use_hier:
            print('use Hiereachial Transformer')
            self.h_layer = Hierachial_Transformer(state_shape, units=units, num_heads=num_heads, drop_rate=0, 
            neighbours=neighbours, make_rotation=make_rotation, time_step=time_step, num_traj=num_traj, 
            random_aug=random_aug, no_ego_fut=no_ego_fut, no_neighbor_fut=no_neighbor_fut, carla=carla)   # state_shape = (6, 10, 5)


        dummy_state = tf.constant(np.zeros(shape=(32,) + state_shape, dtype=np.float32)) # (32, 6, 10, 5)
        mask = tf.ones([32, dummy_state.get_shape()[1]])

        if not bptt:  # Entra
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((32,256))

        if use_map or use_hier:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*2,path_length,5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state, mask=m, init_state=init_state, map_state=map_s)
        self.summary()
    
    
    def call(self, states, mask, test=False, init_state=None, map_state=None, curr_frames=None, aug=True, vision=None):
        """
        states:    [batch, 6, 10, 5]
        vision:    [batch, vision_dim] (ex.: 280 do CNN)
        output:     (features, aux) with features [batch, num_traj, rep_dim]
        """


        if self.use_hier:
            if test:
                # print(states.get_shape(),map_state.get_shape())
                states,val = self.h_layer(states, test, map_state, aug)
                # return states,val
            else:
                states = self.h_layer(states, test, map_state, aug)
        
        if self.use_vision and vision is not None:
            v_emb = self.vision_encoder(vision)
            v_emb = v_emb[:, tf.newaxis, :]  # (batch, 1, rep_dim)
            v_emb = tf.repeat(v_emb, repeats=tf.shape(states)[1], axis=1)
            fused = tf.concat([states, v_emb], axis=-1)
            states = self.vision_fusion(fused)
        
        if self.use_map:
            return states, curr_frames
        
        return states, None