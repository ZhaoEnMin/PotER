import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger
import cv2
import matplotlib.pyplot as plt
import gym

from baselines.common import set_global_seeds, explained_variance
from baselines.common.self_imitation import SelfImitation
from baselines.common.runners import AbstractEnvRunner
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.a2c.utils import EpisodeStats

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            sil_update=4, sil_beta=0.0):

        sess = tf_util.make_session()
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)
        sil_model = policy(sess, ob_space, ac_space, nenvs, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef
        value_avg = tf.reduce_mean(train_model.vf)

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, v_avg, _ = sess.run(
                [pg_loss, vf_loss, entropy, value_avg, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, v_avg

        self.sil = SelfImitation(sil_model.X, sil_model.vf, 
                sil_model.entropy, sil_model.value, sil_model.neg_log_prob,
                ac_space, np.sign, n_env=nenvs, n_update=sil_update, beta=sil_beta)
        self.sil.build_train_op(params, trainer, LR, max_grad_norm=max_grad_norm)
        
        def sil_train():
            cur_lr = lr.value()
            return self.sil.train(sess, cur_lr)

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.sil_train = sil_train
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_raw_rewards = [],[],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            #print(self.obs[0,:,:,0].reshape(7056)[5445])
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, raw_rewards, dones, _ = self.env.step(actions)
            obs,raw_rewards,gelabel=obsaver(obs,raw_rewards,self.gelabel,dones,16)
            
            obs,raw_rewards,gelabel1=obsaver1(obs,raw_rewards,self.gelabel1,dones,16)
            self.gelabel1= gelabel1
            #print(self.gelabel,dones)        
            rewards = np.sign(raw_rewards)
            self.gelabel= gelabel
            self.states = states
            self.dones = dones
            if hasattr(self.model, 'sil'):
                self.model.sil.step(self.obs, actions, raw_rewards, dones)
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
            mb_raw_rewards.append(raw_rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_raw_rewards = mb_raw_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_raw_rewards

    def runset(self,sum_turn_reward):   
        i=0
        k=0
        pi=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        aobslabel=0
        dobslabel=0
        aaobs=[]
        ddobs=[]
        oj=np.zeros(7056)
        mb_obs=np.zeros((16,100000,7056))
        while i <500:#for all env,we test for num turn
            #save obs
            #print(sum_turn_reward)
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            obs, raw_rewards, dones , infos = self.env.step(actions)#infos.size:({'ale.lives':6}*16) ,obs.size:16*84*84*4
            self.sum_reward=self.sum_reward+raw_rewards
            self.dones=dones
            self.states = states
            #print(self.states)
            #print(infos,k)
            k=k+1
            #self.infos = infos
            for e , crewards in enumerate(self.sum_reward ):
                if crewards+0.1>sum_turn_reward and sum(sum((obs[e,:,:,0])))>0:#after get all rewards we have explored/can easily get(eg.>50%)
                    #print(e,pi[e])
                    mb_obs[e,pi[e],:]=obs[e,:,:,0].reshape(7056)
                    mb_obs[e,pi[e]+1,:]=obs[e,:,:,1].reshape(7056)
                    mb_obs[e,pi[e]+2,:]=obs[e,:,:,2].reshape(7056)
                    mb_obs[e,pi[e]+3,:]=obs[e,:,:,3].reshape(7056)
                    pi[e]=pi[e]+4
                    if self.label[e]==0:
                        self.infosbegin[e]=infos[e]
                        self.label[e]=1

            for n, info in enumerate(infos):
                if info!=self.infosbegin[n] and self.sum_reward[n]+0.1>sum_turn_reward:
                    if pi[n]>30:#游戏停止标签
                        self.obs[n] = self.obs[n]*0
                        #self.dones[n]=True#重新开始游戏
                        self.sum_reward[n]=0
                        self.label[n]=0#change the label whether get the known rewards
                        self.pilabel[n]=pi[n]
                        i=i+1#finish one turn of this game
                        print(i,n,aobslabel,aobslabel+pi[n]-30,self.infosbegin,infos)
                        aobs=mb_obs[n,0:(pi[n]-30),:]
                        dobs=mb_obs[n,pi[n]-30:pi[n],:]
                        mb_obs[n,:,:]=np.zeros((100000,7056))
                        aaobs.append(aobs)
                        ddobs.append(dobs)
                        pi[n]=0
                        
                    else:
                        self.obs[n] = self.obs[n]*0
                        #self.dones[n]=True#重新开始游戏
                        self.sum_reward[n]=0
                        self.label[n]=0#change the label whether get the known rewards
                        self.pilabel[n]=pi[n]
                        mb_obs[n,:,:]=np.zeros((100000,7056))
                        #finish one turn of this game
                        #print(n,aobslabel,aobslabel+pi[n]-36,self.infosbegin,infos)
                        pi[n]=0

            self.obs=obs
            
        #batch of steps to batch of rollouts
            
        #aobs=mobs[0:pi-1,:]
        ifnext,obsmin,obsdeath,obslabel,obsaver=getsetgoal(aaobs,ddobs)#ifnext==0 ->this image else next 
        obsdeathlabel=(obsdeath>0)
        np.save('obsmin.npy',obsmin)
        obsmin[obsdeathlabel]=0
        #np.save('obsmin.npy',obsmin)
        np.save('obsdeath.npy',obsdeath)
        np.save('obslabel.npy',obslabel)
        for i in range(7056):
            oj[i]=obsmin[i]*obslabel[i]*obsmin[i]
        for p in range(7056):
            if obsmin[p]==max(obsmin):
                k=p
                break
        for p in range(7056):
            if oj[p]==max(oj):
                k1=p
                break
        return ifnext,k,k1,obsaver


def getsetgoal(aaobs,ddobs):
    chlabel1=np.zeros(7056)
    chlabel=np.zeros(7056)
    obsmin0=np.zeros(7056)
    obsmin1=np.zeros(7056)
    obsdeath1=np.zeros(7056)
    obsdeath=np.zeros(7056)
    ch1labellabel=0
    for j in range(len(aaobs)):#这里假设并不会出现>2的几种画面
        obs=aaobs[j]
        dobs=ddobs[j]
        ch1ch=np.zeros(7056)
        ifnext=0
        for i in range(obs.shape[0]):
            obs1=obs[i,:]#初始状态图
            ju=sum(np.sign(abs(obs[0,:]-obs1)))#different from averset size
            if ju<300 and ifnext==0:#in this env
                d=(abs(obs[0,:]-obs1)>0)#the agent state
                chlabel = chlabel+np.sign(abs(obs[0,:]-obs1))#改变次数
                for k in range(7056):
                    if d[k]:
                        if (i<obsmin0[k] and obsmin0[k]!=0) or obsmin0[k]==0:
                            obsmin0[k]=i
            else:#next env,env have been changed
                ifnext=1
                if ju>300:
                    if ch1labellabel<1 and sum(ch1ch)>0:
                        ch1ch=obs1                      
                        ch1labellabel=ch1labellabel+1
                    elif ch1labellabel<1 and sum(ch1ch)==0:
                        p_i=0
                    else:
                        d=((abs(obs1-ch1ch))>0)
                        chlabel1=chlabel1+np.sign(abs(obs1-ch1ch))
                        for k in range(7056):
                            if d[k]:
                                if (i<obsmin1[k] and obsmin1[k]!=0) or obsmin1[k]==0:
                                    obsmin1[k]=i
        on=obs[0,:]
        onlabel=0
        for ii in range(dobs.shape[0]):
            obs2=dobs[ii,:]
            ju=sum(np.sign(abs(on-obs2)))#
            if ju<300:
                d0=(abs(on-obs2)>0)
                obsdeath[d0]=1
        
            elif ju>299 and onlabel==0:
                on1=obs2
                onlabel=1
                d0=(abs(on1-obs2)>0)
                obsdeath1[d0]=1
            else:
                d0=(abs(on1-obs2)>0)
                ju1=sum(np.sign(abs(on1-obs2)))#
                if ju1<300:
                    obsdeath1[d0]=1
    if ifnext==0:
        return 0,obsmin0,obsdeath,chlabel,obs[0,:]
    else:
        return 1,obsmin1,obsdeath1,chlabel1,ch1ch
   
   

 


def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, sil_update=4, sil_beta=0.0):
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, sil_update=sil_update, sil_beta=sil_beta)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    episode_stats = EpisodeStats(nsteps, nenvs)
    nbatch = nenvs*nsteps
    tstart = time.time()
    
    global ifnext,a,b,obsuse,w
#    ifnext,a,b,obsuse=runner.runset(0)
#    np.save('runner2obsaver.npy',obsuse)
    b=5197
    a=4925
    ifnext=0
    global ifnext1,a1,b1,obsuse1
    obsuse=np.load('runner2obsaver.npy')
    obsuse1=np.load('runner2obsaver.npy')
    b1=5387
    print(ifnext,a,b)
    k=0
    w=0
    mean_r=np.zeros(1100000)
    best_r=np.zeros(1100000)
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, raw_rewards = runner.run()      
#        obs,raw_rewards,gelabel=obsaver(obs,raw_rewards,gelabel,masks) 
#        rewards = np.sign(raw_rewards)
#        if sum(raw_rewards)>99:
#            print(raw_rewards,masks)
        episode_stats.feed(raw_rewards, masks)      
        policy_loss, value_loss, policy_entropy, v_avg = model.train(obs, states, rewards, masks, actions, values)
        sil_loss, sil_adv, sil_samples, sil_nlogp = model.sil_train()
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if 2*update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            #print(values,rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("episode_reward", episode_stats.mean_reward())
            
            logger.record_tabular("best_episode_reward", float(model.sil.get_best_reward()))
            mean_r[k]=episode_stats.mean_reward()
            best_r[k]=float(model.sil.get_best_reward())
            print(episode_stats.mean_reward(),float(model.sil.get_best_reward()))
            k=k+1
            np.save('mean_r.npy',mean_r)
            np.save('best_r.npy',best_r)
            if sil_update > 0:
                logger.record_tabular("sil_num_episodes", float(model.sil.num_episodes()))
                logger.record_tabular("sil_valid_samples", float(sil_samples))
                logger.record_tabular("sil_steps", float(model.sil.num_steps()))
            logger.dump_tabular()
#            if mean_r[k]>0.8 and k>4:#完成很高，基本收敛
#                global ifnext1,a1,b1,obsuse1
#                ifnext1,a1,b1,obsuse1=runner.runset(best_r[k])
#                print(best_r[k])
#                np.save('runner2obsaver1.npy',obsuse1)
#                print(ifnext1,a1,b1)
#                w=w+1
                 
                
    env.close()
    return model


def obsaver(obs,rewards,gelabel,done,nenvs):
    for i in range(nenvs):
        for j in range(4):
            obs1=obs[i,:,:,j].reshape(7056)
            if done[i]:
                gelabel[i]=0
            if obs1[b]==0:
                break
            else:
                if gelabel[i]==0 and obs1[b]-obsuse[b]!=0 and obs1[b]!=230:
                    rewards[i]=2
                    gelabel[i]=1
                elif gelabel[i]==0 and obs1[b]-obsuse[b]==0:
                    obs1[b]=230
                    obs1[b-84]=230
                    obs1[b-168]=230
                    obs1[b-1]=230
                    obs1[b-85]=230
                    obs1[b-169]=230
                    obs[i,:,:,j]=obs1.reshape(84,84)
    return obs,rewards,gelabel



def obsaver1(obs,rewards,gelabel,done,nenvs):
    for i in range(nenvs):
        for j in range(4):
            obs1=obs[i,:,:,j].reshape(7056)
            if done[i]:
                gelabel[i]=0
            if obs1[b1]==0:
                break
            else:
                if gelabel[i]==0 and obs1[b1]-obsuse1[b1]!=0 and obs1[b1]!=230:
                    rewards[i]=2
                    gelabel[i]=1
                elif gelabel[i]==0 and obs1[b1]-obsuse1[b1]==0:
                    obs1[b1]=230
                    obs1[b1-84]=230
                    obs1[b1-168]=230
                    obs1[b1-1]=230
                    obs1[b1-85]=230
                    obs1[b1-169]=230
                    obs[i,:,:,j]=obs1.reshape(84,84)
    return obs,rewards,gelabel
