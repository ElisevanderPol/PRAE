import os
from itertools import count

import gym
import torch

from prae.helpers import get_model, load_network, load_abstract_mdp, \
    state_prep_np, state_prep_torch, get_transitions, make_env
from prae.envs import RoomEnv
from prae.models.policy import Policy
from prae.runner import get_actions


class Evaluator:
    """
    plan-and-eval runner
    """

    def __init__(self, env, args):
        """
        """
        self.env = env
        self.args = args
        self.base_file = os.path.join("runs", args.log_dir, str(args.seed))

    def loop(self, n_epochs=100):
        """
        """
        network = get_model(self.env.action_space.n, self.args)

        returns = []
        lens = []
        # For each training epoch, load the model at that time
        for epoch in range(self.args.start_epochs,
                           self.args.model_train_epochs):

            network = load_network(network, self.base_file, epoch)
            mdp = load_abstract_mdp(self.base_file, epoch, network,
                                    self.args.trans_tau, self.args.gamma)
            # Plan on abstract MDP + evaluate performance
            average_return, avg_len = self.evaluate(network, mdp, epoch)

            returns.append(average_return)
            lens.append(avg_len)
            print(f"Epoch {epoch}, return: {average_return}, length: {avg_len}")

        return returns, lens


    def evaluate(self, network, mdp, epoch):
        """
        """
        returns = []
        lengths = []

        # Average performance over eval_eps episodes
        for episode in range(self.args.eval_eps):
            state = self.env.reset()

            # Plan on abstract MDP
            goal_state = self.env.get_goal_state()
            values, q_values, mdp = plan(network, mdp, goal_state,
                                         self.env.action_space.n, self.args)
            # Store resulting plan + mdp
            itr_folder = os.path.join(self.base_file, str(epoch))
            dump_epoch_mdp(itr_folder, mdp, values, q_values)

            # Policy wrapper that lifts
            policy = Policy(network, q_values, mdp, eps=self.args.hinge)

            rewards = []
            for t in count():
                action = policy.select_action(state, eta=self.args.eta)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                if done:
                    ep_returns = []
                    g = 0
                    for r in reversed(rewards):
                        g = policy.mdp.gamma * g + r
                        ep_returns.append(g)
                    episode_return = ep_returns[-1]
                    returns.append(episode_return)
                    ep_len = len(ep_returns)
                    lengths.append(ep_len)
                    break
        return sum(returns)/len(returns), sum(lengths)/len(lengths)


def plan(network, mdp, goal_state, n_actions, args):
    """
    Plan on the abstract MDP
    """
    device = network.device

    abstract_states = state_prep_torch(mdp.states, device)

    # Reward function
    rewards = torch.zeros_like(mdp.rewards).to(network.device)
    g = state_prep_np(goal_state, device)
    z_g = network.state_encoder(g)
    abstract_states = torch.cat([abstract_states, z_g], dim=0)
    mdp.abstract_states = prune_duplicate_goal(abstract_states,
                                               z_g).squeeze(0)
    rewards = torch.zeros((mdp.abstract_states.shape[0], 1))
    rewards[-1] = 1
    mdp.abstract_rewards = rewards.to(device)

    # Transition function
    actions = torch.from_numpy(get_actions(mdp.abstract_states, network,
                          n_actions)).float().to(network.device)
    mdp.abstract_actions = actions.to(device)
    transitions = get_transitions(mdp.abstract_states,
                                  mdp.abstract_states+actions, network,
                                  args.trans_tau).to(network.device)
    # (Q-)Value function
    values = torch.zeros_like(mdp.abstract_rewards).to(network.device)
    mdp.abstract_transitions = transitions
    for it in range(args.n_sweeps-1):
        q_values = sweep(mdp, values, n_actions)
        values = torch.max(q_values, dim=1)[0].unsqueeze(1)
    q_values = sweep(mdp, values, n_actions)

    return values, q_values, mdp



def sweep(mdp, values, n_actions):
    """
    One sweep of Value Iteration
    """
    q_values = []
    for i in range(n_actions):
        adj = mdp.abstract_transitions[i]
        v_r = mdp.gamma * values
        q_a = torch.mm(adj, v_r) + mdp.abstract_rewards
        q_values.append(q_a)
    q_values = torch.cat(q_values, dim=1)
    return q_values



def prune_duplicate_goal(z_batch, z_goal):
    """
    Ensure goal is in z_batch only once
    """
    l = []
    for idx, z in enumerate(z_batch):
        s = 0.5 * (z - z_goal).pow(2).sum(dim=1)
        if s > 1e-8:
            app_z = z.unsqueeze(0)
            l.append(app_z)
    l.append(z_goal)
    return torch.cat(l, dim=0).unsqueeze(0)


def dump_epoch_mdp(itr_folder, mdp, values, q_values):
    """
    Store MDP components + value function
    """
    s = mdp.abstract_states.detach().cpu().numpy()
    s.dump(os.path.join(itr_folder, f"a_states.npy"))
    r = mdp.abstract_rewards.detach().cpu().numpy()
    r.dump(os.path.join(itr_folder, f"a_rewards.npy"))
    a = mdp.abstract_actions.detach().cpu().numpy()
    a.dump(os.path.join(itr_folder, f"a_actions.npy"))
    v = values.squeeze().detach().cpu().numpy()
    v.dump(os.path.join(itr_folder, f"a_values.npy"))
    q = q_values.squeeze().detach().cpu().numpy()
    q.dump(os.path.join(itr_folder, f"a_qvalues.npy"))
    t = mdp.abstract_transitions.detach().cpu().numpy()
    t.dump(os.path.join(itr_folder, f"a_transitions.npy"))
