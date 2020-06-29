import os
import time

import numpy as np

import torch
import torch.optim as optim

import gym

from prae.policies import UniformPolicy
from prae.data_collector import DataCollector
from prae.helpers import get_data_loaders, to_device, get_model, one_hot, \
    unsqueeze_samples, set_seeds, store_model, prune_duplicates, \
    progress_bar, get_trainer, make_env, get_actions, get_rewards


class Runner:
    """
    Outer loop train runner
    """
    def __init__(self, env, args):
        """
        """
        self.env = env
        self.args = args
        self.base_file = os.path.join("runs", args.log_dir, str(args.seed))

    def loop(self):
        """
        """
        # Collect data
        policy = UniformPolicy(self.env.action_space)
        train_loader, valid_loader = self.collect_data(policy)
        # Initialize model
        network = get_model(self.env.action_space.n, self.args)
        # Train loop
        for itr in range(self.args.model_train_epochs):
            print(f"Train epoch {itr}")
            # Train step
            network = self.train_model(network, train_loader)
            # Validation step
            self.eval_model(network, valid_loader)

            itr_folder = os.path.join(self.base_file, f"{itr}")
            if not os.path.exists(itr_folder):
                os.makedirs(itr_folder)
            # Store network
            torch.save(network.state_dict(), os.path.join(itr_folder,
                                                          f"model.pt"))
            # Store abstract MDP components
            disc_states, rewards, actions = self.make_model(network,
                                                            train_loader)
            store_model(disc_states, rewards, actions, itr_folder)

    def train_model(self, model, data):
        """
        Train iteration
        """
        trainer = get_trainer(self.args, model, train=True)
        model.train(True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=self.args.lr)

        train_loss = 0.
        t0 = time.time()
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training {n_train_params} params")

        for batch_idx, data_batch in enumerate(data):
            #if data_batch[0].shape[0] < self.args.batch_size:
            #    continue
            obs, action, next_obs, r, neg_obs = to_device(data_batch,
                                                          model.device)
            optimizer.zero_grad()

            action_onehot = one_hot(action, self.env.action_space.n)
            loss = trainer(model, obs, action_onehot, next_obs, r, neg_obs)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar(batch_idx, len(data))

        t1 = time.time()
        print("\n", "train loss", train_loss/(batch_idx+1), t1-t0, "s")
        return model

    def eval_model(self, model, data):
        """
        Validation iteration
        """
        trainer = get_trainer(self.args, model, train=False)
        model.train(False)

        valid_loss = 0.
        for batch_idx, data_batch in enumerate(data):
            obs, action, next_obs, r, neg_obs = to_device(data_batch,
                                                          model.device)

            action_onehot = one_hot(action, self.env.action_space.n)
            loss = trainer(model, obs, action_onehot, next_obs, r, neg_obs)

            valid_loss += loss.item()

            progress_bar(batch_idx, len(data))

        print("valid loss", valid_loss/(batch_idx+1))
        return model

    def collect_data(self, policy):
        """
        Collect datasets for learning abstract MDP
        """
        n_ep = self.args.n_episodes
        test_ep = max(n_ep//5, 1)

        data_collector = DataCollector(self.env, policy, self.args.data_folder)

        data_collector.collect(self.args.seed, n_episodes=n_ep, dset="train")
        data_collector.collect(self.args.seed, n_episodes=test_ep,
                               dset="valid")

        train_loader, test_loader = get_data_loaders(data_collector.data_folder,
                                                     self.env.action_space.shape,
                                                     self.args)
        return train_loader, test_loader

    def discretize(self, model, data_set, prune_off=False):
        """
        Sample states to act as prototypes.
        """
        n_total = self.args.batch_size_sample
        tot = 0
        z_stack = []
        for batch_idx, data_batch in enumerate(data_set):
            obs, action, next_obs, r, neg_obs = to_device(data_batch,
                                                          model.device)
            z = model.state_encoder(obs)
            if z.shape[0] > self.args.batch_size_sample:
                z = z[:self.args.batch_size_sample]

            z_stack.append(z)
            tot += data_batch[0].shape[0]

            if tot >= n_total:
                break
        z_stack = torch.cat(z_stack, dim=0)
        if not prune_off:
            pruned_z = prune_duplicates(z_stack).detach().cpu().numpy()
        else:
            pruned_z = z_stack.detach().cpu().numpy()
        return pruned_z


    def make_model(self, network, train_loader):
        """
        Make abstract MDP components
        """
        agg_states = self.discretize(network, train_loader,
                                     prune_off=self.args.prune_off)

        agg_reward = get_rewards(agg_states, network)
        action_embeddings = get_actions(agg_states, network,
                                        self.env.action_space.n)
        return agg_states, agg_reward, action_embeddings
