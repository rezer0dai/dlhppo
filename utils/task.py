from torch.distributions import Normal
from utils.policy import PGDist, NormalWithAction

import config

class Task:
    def __init__(self, RLTask, do_assess=True, goals=None, update_dist=False):
        self.ENV = RLTask()

#        self.goals = goals
        self.do_assess = do_assess
        self.update_dist = update_dist

    def reset(self, agent, seed, learn_mode):
        einfo = self.ENV.reset(agent, seed, learn_mode)
#        if self.do_assess:
#            self.goals = einfo.goals # this can be problematic in HRL settings
            #  self.goals = self.ENV.goals
        return einfo.states

    def goal(self):
        return self.ENV.get_goals()

    def step(self, e_pi, t_pi, learn_mode):
        actions = e_pi.action()
        einfo = self.ENV.step(e_pi.params(False))
        
#        if self.do_assess:
#            self.goals = einfo.goals
#        goods = [True] * len(einfo.rewards)
        goods = einfo.goods

        # temporararely - as we recalc now all rews when sampling
        rewards = einfo.rewards if not learn_mode else einfo.custom_rewards

        # actions reflect finally choosend actions!!
#        pi = e_pi.q_action(einfo.actions)
        pi = einfo.pi

        # BIGGEST UPDATE, QUESTIONABLE TOO -> HAC
#        if self.update_dist and self.ENV.do_update():
#            n = NormalWithAction(einfo.actions, e_pi.dist.scale)
#            pi = PGDist(n).q_action(einfo.actions)
#            assert False
#
        # TODO: KICK OFF
#        if config.HRL_ACTION_SIZE == einfo.actions.shape[-1]: 
#            pi[:, config.HRL_ACTION_SIZE:config.HRL_ACTION_SIZE+4*2] = einfo.actionsZ #TODO
#
        log_prob = t_pi.log_prob(einfo.actions)
        return log_prob, pi, einfo.actions, einfo.states, rewards, einfo.dones, goods
