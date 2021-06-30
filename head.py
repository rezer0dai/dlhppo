import torch

torch.set_default_dtype(torch.float32)

from alchemy.agent import Agent, BrainDescription
from alchemy.env import Env

import model
from utils.task import Task
from utils.encoders import IdentityEncoder, GlobalNormalizer, IEncoder, GoalGlobalNorm, GoalIdentity
from utils.her import HER, CreditAssignment

import torch.nn as nn

import config
import numpy as np
import random, timebudget

import config

action_to_goal_enc = nn.Sequential(
        nn.Linear(config.HRL_ACTION_SIZE, config.INFO_BOTTLENECK_SIZE, bias=False),
        nn.Tanh(),
        nn.Linear(config.INFO_BOTTLENECK_SIZE, config.HRL_GOAL_SIZE),
        nn.Tanh(),
        ).to(config.DEVICE)
action_to_goal_enc.share_memory()

def Actor(action_size, layers, ibottleneck):
    return lambda: model.ActorFactory(
        layers, action_size=action_size, 
        f_mean_clip=torch.tanh,
        f_scale_clip=lambda x: -(1+torch.relu(x)),
        ibottleneck=ibottleneck,
        noise_scale=1.)

HighLevelActor = lambda ec_size: Actor(
    config.HRL_ACTION_SIZE,
    layers = [ec_size+config.CORE_GOAL_SIZE, *config.HI_ARCH, config.HRL_ACTION_SIZE],
    ibottleneck=lambda x: x)

LowLevelActor = lambda ec_size: Actor(
    config.ACTION_SIZE,
    layers = [ec_size+config.HRL_ACTION_SIZE, *config.LO_ARCH, config.ACTION_SIZE],
    ibottleneck=action_to_goal_enc)

def Critic(ec_size):
    return lambda: model.Critic(1, 1, 
            ec_size+config.HRL_GOAL_SIZE, 
            config.ACTION_SIZE, 
            config.HI_ARCH,
            ibottleneck=action_to_goal_enc)

goal_encoder_ex = GoalGlobalNorm(config.CORE_ORIGINAL_GOAL_SIZE).to(config.DEVICE)# if not config.ERGOJR else GoalIdentity(HL_GOAL_SIZE)
goal_encoder_ex.share_memory()
class GlobalNormalizerWithTimeEx(IEncoder):
    def __init__(self, size_in, lowlevel):
        super().__init__(size_in=size_in, size_out=size_in, n_features=1)

        self.lowlevel = lowlevel
        self.enc = GlobalNormalizer(size_in if config.LEAK2LL or not lowlevel else config.LL_STATE_SIZE, 1).to(config.DEVICE)
        self.goal_encoder = goal_encoder_ex

    def forward(self, states, memory):
        states = states.to(config.DEVICE)
        memory = memory.to(config.DEVICE)
        enc = lambda data: self.goal_encoder(data).view(len(data), -1)
        pos = lambda buf, b, e: buf[:, b*config.CORE_ORIGINAL_GOAL_SIZE:e*config.CORE_ORIGINAL_GOAL_SIZE]

        # goal, {current, previous} arm pos 
        arm_pos = pos(states, 1, 3)
        if config.LEAK2LL or not self.lowlevel:# achieved goal and actual goal leaking hint what is our goal to low level
            arm_pos = enc(
                    torch.cat([pos(states, 0, 1), arm_pos, pos(states, -1, 10000)], 1)
                    )[:, :-config.CORE_ORIGINAL_GOAL_SIZE] # skip goal from here, just add hint via norm, also achieved one will be not used by NN
        else:
            arm_pos = torch.cat([pos(arm_pos, 0, 1),
                enc( arm_pos )
                ], 1)# first arm_pos aka fake achieved will be not used anyway

        obj_pos_w_goal = pos(states, -(2*config.PUSHER+1), 10000)
        if config.PUSHER and not self.lowlevel: # object pos, prev pos, goal --> only high level
            obj_pos_w_goal = enc(obj_pos_w_goal)

        state = states
        if self.lowlevel and not config.LEAK2LL:
            state = states[:, config.CORE_ORIGINAL_GOAL_SIZE:][:, :config.LL_STATE_SIZE]

        encoded, memory = self.enc(state, memory)

        if self.lowlevel and not config.LEAK2LL:
            encoded = torch.cat([states[:, :config.CORE_ORIGINAL_GOAL_SIZE], encoded, states[:, config.CORE_ORIGINAL_GOAL_SIZE+config.LL_STATE_SIZE:]], 1)

        encoded = pos(encoded, 3, -(2*config.PUSHER+1)) # skip object + goal positions, those will be added by goal_encoder

        # note : achievedel goal, and goal will be skipped from NN ( this norm is used in encoders just for puting it trough NN )
        encoded = torch.cat([arm_pos, encoded, obj_pos_w_goal], 1)

        if states.shape[-1] != encoded.shape[-1]:
            print("\nshappezzz:", states.shape[-1], encoded.shape[-1], self.lowlevel, obj_pos_w_goal.shape)
        assert states.shape[-1] == encoded.shape[-1]
        return encoded, memory
            
from utils.memory import Memory
from utils.memlocal import MemoryBoost

from tasks.oaiproc import make_env

def new_agent(
    credit_assign, brains, lr_critic,
    actor, critic, goal_encoder, encoder, freeze_delta, freeze_count, 
    action_size, n_rewards, n_step, max_steps,
    n_actors, detach_actors, n_critics, detach_critics, 
    stable_probs,
    goal_size, state_size,
    good_reach=1, model_path="checkpoints", save=False, load=False,
    eval_delay=60, eval_limit=10, dbg=False,
    recalc_per_push=None, fast_fail=False
    ):

    goal_encoder.share_memory()
    encoder.share_memory()

    # g, s, f, a, p, r, n_g, n_s, n_f, c, d
    memory = Memory(brains[0].memory_size, recalc_feats_delay=10000, chunks=[
        goal_size, state_size, encoder.features_n(), 3*action_size, action_size, 
        n_rewards, goal_size, state_size, encoder.features_n(), n_rewards, 1], 
                    ep_draw=3, ep_dream=config.MIN_N_SIM // 2)

    experience = lambda descs, brain: MemoryBoost(descs, memory, credit_assign, brain, good_reach, recalc_per_episode=1, recalc_per_push=recalc_per_push, device=config.DEVICE)

    agent = Agent(
        config.DEVICE,
        brains, experience,
        Actor=actor, Critic=critic, 
        goal_encoder=goal_encoder, encoder=encoder, 
        n_agents=1,
        n_actors=n_actors, detach_actors=detach_actors, n_critics=n_critics, detach_critics=detach_critics, 
        stable_probs=stable_probs,
        resample_delay=3, min_step=1,
        state_size=state_size, action_size=action_size,
        freeze_delta=freeze_delta, freeze_count=freeze_count,
        lr_critic=lr_critic, clip_norm=1., q_clip=2e-1,
        model_path=model_path, save=save, load=load, delay=10,
        min_n_sim=config.MIN_N_SIM,
        loss_callback=None,#loss_callback,
    )

    # defined above
    env = Env(agent, 
            total_envs=config.TOTAL_ENV, n_history=1, history_features=encoder.features_n(), state_size=encoder.in_size(),
            n_step=n_step, send_delta=max_steps,
            eval_limit=eval_limit, eval_ratio=.5, max_n_episode=max_steps, eval_delay=eval_delay,
            mcts_random_cap=100000, mcts_rounds=1, mcts_random_ratio=10, limit=100000,
            debug_stats=dbg, fast_fail=fast_fail)
    
    return agent, env
    
def install_lowlevel(low_level_task, Her):
    KEYID = config.PREFIX+"_ll"
    RECALC_PER_PUSH = 20
    LL_GOAL_SIZE = config.HRL_ACTION_SIZE
    LL_STATE_SIZE = config.CORE_STATE_SIZE + config.CORE_ORIGINAL_GOAL_SIZE
    LL_MAX_STEPS = 1 + config.HRL_HIGH_STEP * config.HRL_STEP_COUNT

    goal_encoder = GoalIdentity(LL_GOAL_SIZE)
    ll_state_encoder = GlobalNormalizerWithTimeEx(LL_STATE_SIZE, True)# if not config.ERGOJR else IdentityEncoder(HL_STATE_SIZE)
    state_encoder = ll_state_encoder#IdentityEncoder(LL_STATE_SIZE)

    senv = make_env(False, config.ENV_NAME)
    for seed in range(1):#00):
        do_sample(senv, seed, goal_encoder, state_encoder)

    delay = 10
    repeat = 10
    optim_n = 1

    brain = [
            BrainDescription(
                memory_size=1 * config.MIN_N_SIM * LL_MAX_STEPS, batch_size=config.LL_BATCH_SIZE,
                optim_pool_size = 1 * config.MIN_N_SIM * RECALC_PER_PUSH * config.HRL_HIGH_STEP,
                optim_epochs=repeat // optim_n, optim_batch_size=2*config.LL_BATCH_SIZE, recalc_delay=7,
                lr_actor=3e-4, learning_delay=delay, learning_repeat=repeat,
                warmup = 0,
                sync_delta_a=3, sync_delta_c=2, tau_actor=5e-2, tau_critic=5e-2,
                bellman=False, ppo_eps=2e-1, natural=False, mean_only=False, separate_actors=False),
    ]
    print("\nLOW LEVEL POLICY: \n", [b for b in brain])

    credit_assign = [ Her(her_delay=0,
        cind=0, gae=config.GAE, n_step=config.HRL_LOW_N_STEP, floating_step=config.FLOATING_STEP, 
        gamma=config.GAMMA, gae_tau=.95, 
        her_select_ratio = config.HER_RATIO, resampling=False, kstep_ir=False, clip=2e-1) ]
    
    agent, env = new_agent(
        credit_assign, brain, lr_critic=3e-4,
        actor=LowLevelActor(state_encoder.out_size()), critic=Critic(state_encoder.out_size()),
        goal_encoder=goal_encoder, encoder=state_encoder, freeze_delta=3, freeze_count=3, 
        action_size=config.ACTION_SIZE, n_rewards=1, n_step=config.HRL_LOW_N_STEP, max_steps=LL_MAX_STEPS,
        n_actors=config.N_LL_ACTORS, detach_actors=False, n_critics=config.N_CRITICS, detach_critics=config.DETACH_CRITICS, 
        stable_probs=True,
        goal_size=LL_GOAL_SIZE, state_size=LL_STATE_SIZE,
        good_reach=LL_MAX_STEPS, model_path=KEYID+"_checkpoints", save=config.SAVE, load=config.LOAD,
        eval_delay=None, eval_limit=1, dbg=False,
        recalc_per_push=RECALC_PER_PUSH,
        )

# TEMPORARERLY for hrl.py to allow dream with high level policy
    config.AGENT.append(agent)#.insert(0, agent)

    class LLFetch(Task):
        def goal_met(self, _total_reward, _last_reward):
            return False
    task = LLFetch(lambda : low_level_task)

    return env, task
    
class ReacherCreditAssignment(CreditAssignment):
    def update_goal(self, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps, allowed_mask):
        return ( rewards, goals, states, n_goals, n_states, allowed_mask )
    
import gym
def do_sample(env, seed, goal_encoder, encoder):
    env.seed(seed)
    obs = env.reset()

    for _ in range(config.HRL_HIGH_STEP * config.HRL_STEP_COUNT):
        obs, _, done, _ = env.step(env.action_space.sample())
        if done: 
            break

        positions = [
                obs['desired_goal'],
                obs['achieved_goal'],
                ]

        state = np.concatenate([positions[random.randint(0, 1)], obs['observation'], positions[random.randint(0, 1)]])
        encoder(torch.from_numpy(state).view(1, -1).float().to(config.DEVICE), torch.zeros(1).to(config.DEVICE))
    
def install_highlevel(high_level_task, keyid):
    HL_GOAL_SIZE = config.CORE_GOAL_SIZE
    HL_STATE_SIZE = config.CORE_STATE_SIZE + config.CORE_ORIGINAL_GOAL_SIZE
    HL_MAX_STEPS = config.HRL_HIGH_STEP
    RECALC_PER_PUSH = 20

    # if ergoJR it is already normalized goal!
#    goal_encoder = GoalGlobalNorm(HL_GOAL_SIZE)# if not config.ERGOJR else GoalIdentity(HL_GOAL_SIZE)
    goal_encoder = goal_encoder_ex
    hl_state_encoder = GlobalNormalizerWithTimeEx(HL_STATE_SIZE, False)# if not config.ERGOJR else IdentityEncoder(HL_STATE_SIZE)
    state_encoder = hl_state_encoder#GlobalNormalizerWithTime(goal_encoder, HL_STATE_SIZE, 1)# if not config.ERGOJR else IdentityEncoder(HL_STATE_SIZE)

    delay = 1 * HL_MAX_STEPS
    repeat = 2 * HL_MAX_STEPS

# dekay giving one round for low level policy to adapt
    brain = [
            BrainDescription( # master :: PPO
                memory_size=1 * config.MIN_N_SIM * HL_MAX_STEPS, batch_size=config.HL_BATCH_SIZE,
                optim_pool_size = 1 * config.MIN_N_SIM * RECALC_PER_PUSH * HL_MAX_STEPS,
                optim_epochs=1, optim_batch_size=2*config.HL_BATCH_SIZE, recalc_delay=3,
                lr_actor=1e-4, learning_delay=delay, learning_repeat=repeat,
                warmup = 0,#110,
                sync_delta_a=3, sync_delta_c=2, tau_actor=7e-2, tau_critic=5e-2,
                bellman=False, ppo_eps=2e-1, natural=False, mean_only=False, separate_actors=False),
    ]

    print("\nHIGH LEVEL: \n", [b for b in brain])

    credit_assign = [ ReacherCreditAssignment(
        cind=0, gae=config.GAE, n_step=config.HRL_HIGH_N_STEP, floating_step=config.FLOATING_STEP, 
        gamma=config.GAMMA, gae_tau=.95,
        resampling=False, kstep_ir=True, clip=2e-1) ]

    agent, env = new_agent(
        credit_assign, brain, lr_critic=1e-4,
        actor=HighLevelActor(state_encoder.out_size()), critic=lambda: None,
        goal_encoder=goal_encoder, encoder=state_encoder, freeze_delta=3, freeze_count=3, 
        action_size=config.HRL_ACTION_SIZE, n_rewards=1, n_step=config.HRL_HIGH_N_STEP, max_steps=HL_MAX_STEPS,
        n_actors=config.N_HL_ACTORS, detach_actors=False, n_critics=config.N_CRITICS, detach_critics=config.DETACH_CRITICS, 
        stable_probs=True,
        goal_size=HL_GOAL_SIZE, state_size=HL_STATE_SIZE,
        good_reach=HL_MAX_STEPS, model_path=keyid+"_checkpoints", save=config.SAVE, load=config.LOAD,
        eval_delay=4, eval_limit=10, dbg=True,
        recalc_per_push=RECALC_PER_PUSH)

    config.AGENT.append(agent)#.insert(0, agent)

    senv = make_env(False, config.ENV_NAME)
    for seed in range(1):#00):
        do_sample(senv, seed, goal_encoder, state_encoder)
    goal_encoder.stop_norm()

    class HLFetch(Task):
        def goal_met(self, total_reward, last_reward):
            return last_reward > -config.HRL_STEP_COUNT / 2.
            if config.ERGO_REACHER:
                if last_reward < -3.:
                    return False
                return total_reward > -config.HRL_STEP_COUNT * config.HRL_HIGH_STEP / 3.

            return 0 == last_reward[0] # quite though for bigger low step policy to manage

    task = HLFetch(lambda : high_level_task)
    return env, task
