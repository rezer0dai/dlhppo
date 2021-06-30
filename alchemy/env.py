from __future__ import print_function
import torch
import random
import numpy as np

from collections import deque

class Env:
    def __init__(self, agent,
            total_envs, n_history, history_features, state_size,
            n_step, send_delta,
            eval_limit, eval_ratio, max_n_episode, eval_delay,
            mcts_random_cap, mcts_rounds, mcts_random_ratio, limit,
            debug_stats=True, fast_fail=True, do_not_eval_exploit=False
            ):

        assert n_step <= send_delta, "for GAE, but we apply it in general, we need to send at least n_step samples ( adjust send_delta accordingly! )"

        self.debug_stats = debug_stats
        self.fast_fail = fast_fail
        self.do_not_eval_exploit = do_not_eval_exploit

        self.agent = agent

        # configs
        self.total_envs = total_envs
        self.state_size = state_size // n_history
        self.n_history = n_history
        self.history_features = history_features
        self.send_delta = send_delta
        self.n_step = n_step

        self.max_n_episode = max_n_episode
        self.eval_limit = eval_limit
        self.eval_ratio = eval_ratio
        self.eval_delay = eval_delay
        self.mcts_random_cap = mcts_random_cap
        self.mcts_rounds = mcts_rounds
        self.limit = limit

        # debug state
        self.ep_count = 0
        self.step_count = 0
        self.score = 0
        self.best_max_step = 0
        self.best_min_step = max_n_episode

        self.mcts_random_ratio = mcts_random_ratio
        self.seeds = []
        self.scores = [0]

    def _select_seed(self, n_iter):
        if self.mcts_random_ratio and 0 == n_iter % self.mcts_random_ratio:
            return random.randint(0, len(self.seeds)-1)

        self.seeds.append(np.asarray([
            random.randint(0, self.mcts_random_cap
                ) for _ in range(self.total_envs) ]).ravel())

        return -1

    def start(self, task, callback):
        finished, test_scores = self._evaluate(task)
        if finished:
            return test_scores

        for _ in range(self.limit):
            self.do_episode(task, self.mcts_rounds)

            finished, test_scores = self._evaluate(task) if 0 == len(self.scores) % self.eval_delay else (False, None)

            callback(self.agent, task, test_scores, self.scores[-1], [0], len(self.scores)) # back to user

            if finished:
                break

        return self.scores

    def do_episode(self, task, mcts_rounds):
        local_seeds = [ self.seeds[self._select_seed(len(self.scores))] ] * mcts_rounds
        score = self._simulate(task, local_seeds)
        self.scores.append(torch.stack(score).mean(1).mean(1).sum())
        return score

    def _simulate(self, task, seeds):
        for data, r in self._simulate_steped(task, seeds):
            if r is not None:
                return r

    def step(self, task, seed, n_steps):
        cnt = 0
        acr = 0
        for data, r in self._simulate_steped(task, [seed]):
            cnt += 1
            if data is not None:
                acr += data[6]
            if 0 != cnt % n_steps:
                continue
            yield data, acr
            acr = 0

    def _evaluate(self, task):
        return next(self.evaluate(task, None))

    def evaluate(self, task, n_steps):
        scores = [[] for _ in range(self.agent.n_targets)]
        success = [[] for _ in range(self.agent.n_targets)]
        for z in range(self.eval_limit):
            seed = self.seeds[self._select_seed(len(self.scores))]
            for i in range(len(scores)):
                cnt = 0
                acr = 0
                for data in self._learning_loop(task, *self._history(), seed, False, i):
                    acr += data[6]
                    if n_steps is None:
                        continue
                    cnt += 1
                    if 0 != cnt % n_steps:
                        continue
                    yield data, acr
                    acr = 0

                scores[i].append(self.score.detach().cpu())
                success[i].append(True if task.goal_met(self.score.detach().cpu(), data[-2][0][0].detach().cpu()) else False) # push last reward

            succ = max(np.mean(success, 1))
            if succ < self.eval_ratio:
                if self.fast_fail:
                    break

        if self.eval_limit > 1:
            print("{}\nSUCCES RATIO {}% [mean={}]\n{}".format("="*30, 100. * succ, np.mean(scores), "="*30))
        if succ > self.eval_ratio:
            print("\n environment solved! ", np.mean(scores), [np.mean(scores, 1)])
            print(scores)

# well ...
        yield succ > self.eval_ratio, scores

    def _learning_loop(self, task, f_pi, history, seed, learn_mode, tind=0):
        self.score = 0
        next_state = task.reset(self.agent, seed, learn_mode)
        steps = 0
        while True:
            steps += 1
            #  if not learn_mode and 50 < steps:
            #      break
#            if learn_mode and steps >= self.max_n_episode:
            if steps > self.max_n_episode:
                break

            state = next_state.view(len(history), -1)
            for i, s in enumerate(state):
                history[i].append(s.view(-1).cpu().numpy())
            state = np.asarray(history).reshape(len(history), -1)
            state = torch.from_numpy(state).float()

            goal = task.goal()

            if learn_mode or self.do_not_eval_exploit:
                e_pi, f_pi, t_pi = self.agent.explore(goal, state, f_pi, self.step_count)
            else:
                e_pi, f_pi, t_pi = self.agent.exploit(goal, state, f_pi, tind)

            data = task.step(e_pi, t_pi, learn_mode)
            if data is None:
                break

            log_prob, pi, action, next_state, reward, done, good = data

#            if action.shape[1] == 3: print("-->",steps, self.max_n_episode, learn_mode)
            #  print("-->",steps, self.max_n_episode, learn_mode)

            yield (log_prob, pi, goal, state, f_pi, action, reward, good)

            self.score += reward.mean()
            if sum(done):
                break

        # last state tunning
        state = next_state.view(len(history), -1)
        for i, s in enumerate(state):
            history[i].append(s.view(-1).cpu().numpy())
        state = np.asarray(history).reshape(len(history), -1)
        state = torch.from_numpy(state).float()#.view(len(history), -1)

        # last dummy state -> for selfplay, no need action, reward, goo
        # we need goal + history + state
        yield (
                log_prob, pi, # will not be used
                task.goal(), state, f_pi,
                action,
                reward, # will be used at task.goal_met evaluation possibly..
                [False] * len(good)) # no need to self-play from this state

    def _simulate_steped(self, task, seeds, learn_mode=True):
        self.learn_mode = learn_mode

        scores = []
        while len(scores) != len(seeds):
            e, seed = len(scores), seeds[len(scores)]

            self.ep_count += 1

            goals = []
            states = []
            features = []
            actions = []
            probs = []
            rewards = []
            goods = []

            f_pi, history = self._history()
            features += [f_pi] * 1

            last = 0
            for data in self._learning_loop(task, f_pi, history, seed, learn_mode):
                self.step_count += 1

                log_prob, pi, goal, state, f_pi, action, reward, good = data

                if not len(scores): # only first round of mcts we want to learn!
                    self.agent.step(self.step_count)

                actions.append(pi)
                probs.append(log_prob)
                rewards.append(reward)
                goals.append(goal)
                states.append(state)
                goods.append(good)

                temp = self._share_experience(e, len(states), last)
                if temp != last:
                    exp_delta = (self.send_delta + 2*self.n_step) if last else len(goals)
                    self._share_imidiate(
                            goals[-exp_delta:-self.n_step],
                            states[-exp_delta:-self.n_step],
                            features[-exp_delta:-self.n_step],
                            actions[-exp_delta:-self.n_step],
                            probs[-exp_delta:-self.n_step],
                            rewards[-exp_delta:-self.n_step],
                            goods[-exp_delta:-self.n_step])
                last = temp

                features.append(f_pi)

                # debug
                if sum(good): self._print_stats(e, rewards, action)

                yield data, None

            scores.append(True)

            self._share_final(
                    goals[last:],
                    states[last:],
                    features[last:-1],
                    actions[last:],
                    probs[last:],
                    rewards[last:],
                    goods[last:])

            self.best_max_step = max(self.best_max_step, len(rewards))
            self.best_min_step = min(self.best_min_step, len(rewards))

        yield None, rewards

    def _history(self):
        f_pi = torch.zeros([self.total_envs, self.history_features])
        history = [ deque(maxlen=self.n_history) for _ in range(self.total_envs) ]
        for s in np.zeros([self.n_history, self.state_size]):
            for i in range(len(history)):
                history[i].append(s.ravel())
        return f_pi, history

    def _share_experience(self, e, total, last):
        delta = e + total
        if (delta - self.n_step) % self.send_delta:
            return last# dont overlap
        if total < self.n_step * 3:
            return last# not enough data
        return total - 2 * self.n_step

    def _share_final(self,
            goals, states, features, actions, probs, rewards,
            goods):

#        if len(goals) < self.n_step:
#            return # ok this ep we scatter

        self._share_imidiate(
            goals, states, features, actions, probs, rewards,
            goods, finished=True)

    def _share_imidiate(self,
            goals, states, features, actions, probs, rewards,
            goods, finished=False):

        if not self.learn_mode:
            return
# just wrapper
        self.agent.save(
            goals, states, features, actions, probs, rewards,
            goods, finished)

    def _print_stats(self, e, rewards, action):
        if not self.debug_stats:
            return
        print("\r[{:5d}>{:6d}::{:2d}] steps = {:4d}, max_step = {:3d}/{:3d}, reward={:2f} <action={}...>{}".format(
            self.ep_count, self.step_count, e, len(rewards),
            self.best_max_step, self.best_min_step,
            self.score, action.view(self.total_envs, -1)[0].view(-1)[:3], " "*20), end="")
