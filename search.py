import random, os
def pick(data):
    return random.choice(data)

search_grid = {
    "floating_step" : [True],#, False],
    "goal_extended" : [0, 3],
    "steps" : [70],
    "hrl" : [1, 2, 2],#, 2, 1, 1],
    "action_size" : [10, 20, 40],
    "reward_delta" : [1.],#, .1, 0.],
    "reward_magnitude" : [.01],#, 1., .1],
    "redo" : [False],#True
    "gae" : [True, False, True],
    "batch_sizes" : [4096, 2*4096],
    "SIGMOID" : [True, False],
    "BPO" : [False, True, False],
    "TEST_ENVS" = [["FetchPush-v1"],["FetchPush-v1", "FetchReach-v1", "FetchPush-v1"],  ["FetchPush-v1", "FetchReach-v1"]]
}


config = """
ENV_NAME = "FetchPush-v1"#"FetchReach-v1"#"FetchPickAndPlace-v1"#"FetchSlide-v1"#
CORE_STATE_SIZE = 25#-3#10

FLOATING_STEP = {}

CORE_ORIGINAL_GOAL_SIZE = 3
CORE_GOAL_SIZE = CORE_ORIGINAL_GOAL_SIZE+{}

HRL_HIGH_STEP = {}
HRL_STEP_COUNT = {}
HRL_ACTION_SIZE = {}

HER_RATIO = .3

HRL_LOW_N_STEP = 2 if not FLOATING_STEP else int(HRL_HIGH_STEP * HRL_STEP_COUNT * HER_RATIO / 2)
HRL_HIGH_N_STEP = 4 if not FLOATING_STEP else (HRL_HIGH_STEP // 5 * 3)

HRL_ACTION_TEST_RATIO = .5

MIN_N_SIM = 30
TOTAL_ENV = 1*MIN_N_SIM
DEVICE = "cpu"

REWARD_DELTA = {}
REWARD_MAGNITUDE = {}

REDO = {}

TOTAL_ROUNDS = 200
PREFIX="BPO_HYPER_PARAM_SEARCH"

GAE = {}
BATCH_SIZE = {}
SIGMOID = {}
BPO = {}
TEST_ENVS = {}

"""

values = []
for param in search_grid:
    value = pick(search_grid[param])


    if "hrl" == param:
        values[-1] = values[-1] // value

    values.append(value)

cfg = config.format(*values)

with open("config.py", "w") as f:
    f.write(cfg)

os.system("python hrlex.py")

