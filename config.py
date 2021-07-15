COLAB = True#False#

LOAD = False#True#
SAVE = False#True

DDPG = True#False#

DOUBLE_LEARNING = False#True# ##### False in second DoubleL because EXPLORER only will be used!!!
DL_EXPLORER = True#False#
NORMALIZE = True#False#
LLACTOR_UNOMRED = False#True#
CRITIC_UNORMED = False#True#
TF_LOW = True#None#
BLIND = True#False#
NO_GOAL = False#True#
SELECT_EXP = False#True#
LEAK2LL = False#True#

TIMEFEAT = True#False#
GAMMA = 99# if TIMEFEAT else .85

PANDA = True#False#
ERGOJR = False#True#
MUJOCO = False#True#

BACKLASH = False

assert MUJOCO + PANDA + ERGOJR == 1

CORE_ORIGINAL_GOAL_SIZE = 3

PUSHER = False#True#

if ERGOJR: # no gripper, velo per joint ( #of joints == action_size )
    ACTION_SIZE = 3 + (not PUSHER) * 1#3
    LL_STATE_SIZE = CORE_ORIGINAL_GOAL_SIZE * 2 + ACTION_SIZE * 2 + TIMEFEAT
    CORE_STATE_SIZE = CORE_ORIGINAL_GOAL_SIZE + LL_STATE_SIZE + 3*CORE_ORIGINAL_GOAL_SIZE*PUSHER
else: # arm pos, arm prev pos, arm velo, gripper pos + velo + velp
    ACTION_SIZE = 3 + MUJOCO
    LL_STATE_SIZE = CORE_ORIGINAL_GOAL_SIZE * 3 + 4 * MUJOCO + TIMEFEAT
    CORE_STATE_SIZE = CORE_ORIGINAL_GOAL_SIZE + LL_STATE_SIZE + 6*CORE_ORIGINAL_GOAL_SIZE*PUSHER# velp + gripper, object velp for pusher

ENV_NAME = ("PUSHER" if PUSHER else "REACHER") + "_" + ("ERGOJR" if ERGOJR else ("PANDA" if PANDA else "MUJOCO"))

FLOATING_STEP = True#False#

CORE_GOAL_SIZE = CORE_ORIGINAL_GOAL_SIZE

HRL_HIGH_STEP = 24#10#25#
HRL_STEP_COUNT = 2#10#2#
HRL_ACTION_SIZE = 64#8
INFO_BOTTLENECK_SIZE = 16#32
HRL_GOAL_SIZE = 10#4

HER_RATIO = .6#.5#.4#

HRL_LOW_N_STEP = 2#int(HRL_HIGH_STEP * HRL_STEP_COUNT * (1.-HER_RATIO) / 4)
HRL_HIGH_N_STEP = 5#*HRL_HIGH_STEP//2#40#20#HRL_HIGH_STEP // 10 * 8

HRL_ACTION_TEST_RATIO = None#.15#1.#
HRL_HINDSIGHTACTION_HORIZON = HRL_HIGH_STEP * 10#40#100#

MIN_N_SIM = 20#40#100#
TOTAL_ENV = MIN_N_SIM#(1 + PUSHER)*MIN_N_SIM
DEVICE = "cpu"

REWARD_DELTA = 0.#.5#
REWARD_MAGNITUDE = 1.#2.#

REDO = False

TOTAL_ROUNDS = 300000
PREFIX="multiprocess_220_"+ENV_NAME
# CHANGES : policy.py diff * .5, ac.py probs + (old - new) -> w/o discount, HRL_HIROZON * 50 -> now w/o 50

GAE = True
RECALC_PER_PUSH_LL = (3 if not FLOATING_STEP else 5)
HL_BATCH_SIZE = 2 * MIN_N_SIM * HRL_HIGH_STEP * RECALC_PER_PUSH_LL#learn ppo every episode
LL_BATCH_SIZE = 2 * 2 * MIN_N_SIM*2#200
SIGMOID = False#True#
BPO = False
TEST_ENVS = [ENV_NAME]#, ENV_NAME, "FetchReach-v1"]#"FetchPush-v1","FetchPush-v1"]#"FetchReach-v1"]#"FetchPush-v1"]#, "FetchPush-v1", "FetchReach-v1", "FetchPush-v1"]#

HI_ARCH = [256]*3#400, 300]#
LO_ARCH = [256]*3
RELU = True#False#
WD = 1e-3

N_CRITICS = 2#1#
DETACH_CRITICS = False#True#
N_HL_ACTORS = 1
N_LL_ACTORS = 1

AGENT = []
