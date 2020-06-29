import os
import numpy as np
import gym
import torch
from torch.utils import data
import torch.nn.functional as F

from prae.dataset import StateTransitionsDataset
from prae.models import *
from prae.envs import RoomEnv, CartpoleWrapper, FashionMnistTranslate2dEnv
from prae.distances import HingedSquaredEuclidean
from prae.losses import Loss
from prae.trainers import ActionEquivariantTrainer


def make_env(args):
    """
    Get appropriate env
    """
    train_goals = not args.test_goals
    train_set = not args.test_set
    if args.env == "room":
        env = RoomEnv(n_objects=args.objects, seed=int(args.seed),
                      train=train_goals, random_goal=args.test_set)
    elif args.env == "fashion_translations":
        import torchvision.datasets as datasets
        # from fashion_translate import FashionMnistTranslate2dEnv
        if train_goals:
            goal_mode = 'train'
        else:
            goal_mode = 'test'
        mnist = datasets.FashionMNIST(root='./data/fashion_data',
                                      train=train_set,
                                      download=True, transform=None)

        data = [(digit, label) for (digit, label) in mnist]

        env = FashionMnistTranslate2dEnv(train=train_set, goal_mode=goal_mode,
                                         data=data, seed=int(args.seed))
    elif args.env == "cartpole":
        env = CartpoleWrapper(seed=int(args.seed))
    return env



def get_trainer(args, model, train=True):
    """
    Get wrapper that trains network
    """
    if args.n_neg_samples  == 0:
        neg = False
    elif args.n_neg_samples > 0:
        neg = True
        loss_function = Loss(hinge=args.hinge, neg=neg)
        trainer = ActionEquivariantTrainer(loss_function, train=train)
    return trainer


def set_seeds(seed):
    """
    """
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_prep_np(state, device):
    """
    """
    state = torch.from_numpy(state).to(device).float()
    if len(state.shape) == 3:
        state = state.unsqueeze(0)
    return state


def state_prep_torch(state, device):
    """
    """
    state = state.float().to(device)
    return state


def tile_z(z, n):
    """
    """
    return z.expand(n, z.shape[1])


def get_data_loaders(data_folder, n_actions, args):
    """
    Retrieve train and validation loaders
    """
    train_loader = get_data_loader(True, data_folder, n_actions, args)
    test_loader = get_data_loader(False, data_folder, n_actions, args)
    return train_loader, test_loader


def get_data_loader(train, data_folder, n_actions, args, num_workers=4):
    """
    Get train or validation data set
    """
    dset = "train" if train else "valid"

    path = os.path.join(data_folder, str(args.seed), dset)
    dataset = StateTransitionsDataset(data_loc=path,
                                      n_neg_samples=args.n_neg_samples,
                                      num_actions=n_actions)
    # Ensure different samples for each worker
    seeder = lambda seeder: np.random.seed(int(torch.initial_seed())%(2**32-1))
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=num_workers,
                             worker_init_fn=seeder, pin_memory=False)
    return loader


def to_device(data_tuple, device):
    """
    Move data to device
    """
    device_tuple = [x.to(device) for x in data_tuple[:-2]]
    if data_tuple[-2] is not None:
        device_tuple.append(data_tuple[-2].to(device))
    else:
        device_tuple.append(None)
    if data_tuple[-1] is not None:
        device_tuple.append(data_tuple[-1].to(device))
    else:
        device_tuple.append(None)

    return device_tuple


def get_model(num_actions, args):
    """
    Get network consisting of state encoder, action encoder and reward
    predictor
    """
    state_encoder = get_state_encoder(args.z_dim, args.env, num_actions)
    action_encoder = ActionEncoder(args.z_dim, num_actions)
    rewards = RewardPredictor(args.z_dim)


    model = Model(state_encoder, action_encoder, rewards)

    if not args.cpu:
        model = model.cuda()
    return model


def load_network(network, base_file, epoch):
    """
    Load network weights
    """
    path = os.path.join(base_file, f"{epoch}", "model.pt")
    network.load_state_dict(torch.load(path))
    network.eval()
    return network


def load_abstract_mdp(base_file, epoch, network, tau, gamma):
    """
    Construct plannable abstract MDP
    """
    base_file = os.path.join(base_file, f"{epoch}")
    states = np.load(os.path.join(base_file,
                                  f"sampled_states.npy"),
                     allow_pickle=True)
    rewards = np.load(os.path.join(base_file,
                                   f"sampled_rewards.npy"),
                      allow_pickle=True)
    actions = np.load(os.path.join(base_file,
                                   f"sampled_actions.npy"),
                      allow_pickle=True)

    torch_states = prune_duplicates(torch.from_numpy(states))
    torch_actions = torch.from_numpy(get_actions(torch_states, network,
                                                 actions.shape[0]))
    torch_rewards = torch.from_numpy(get_rewards(torch_states, network))
    transitions = get_transitions(torch_states, torch_states+torch_actions,
                                  network, tau)

    mdp = MDP(torch_states, transitions, torch_rewards, gamma)
    return mdp


def get_rewards(latent_states, network):
    """
    Get rewards for all states
    """
    device = network.device
    torch_latents = torch.tensor(latent_states).float().to(device)
    rewards = network.reward(torch_latents).detach().cpu().numpy()
    return rewards


def get_actions(latent_states, network, n_actions):
    """
    Get action embeddings for all state, action pairs
    """
    device = network.device
    torch_latents = torch.tensor(latent_states).float().to(device)
    actions = []
    batch_size = latent_states.shape[0]
    for action in range(n_actions):
        action_in = torch.ones(batch_size, device=device).long() * action
        action_onehot = one_hot(action_in, n_actions)
        action_embedding = network.action_encoder(torch_latents,
                                                   action_onehot)
        actions.append(action_embedding.unsqueeze(0))
    action_stack = torch.cat(actions, dim=0).detach().cpu().numpy()
    return action_stack


def prune_duplicates(z_batch):
    """
    Remove duplicate latent states
    """
    l = []
    indices = []
    for idx, z in enumerate(z_batch):
        z = z.unsqueeze(0)
        if not in_list(l, z):
            l.append(z)
            indices.append(idx)
    full_list = torch.cat(l, dim=0)
    return full_list


def in_list(l, z):
    """
    check if z is in list l
    """
    for z_p in l:
        s = (z_p - z).pow(2).sum(dim=1)
        if s == 0:
            return True

    return False


def get_transitions(torch_states, torch_actions, network, tau):
    """
    Get softmax over next states
    """
    distance = HingedSquaredEuclidean()
    n_states, state_dim = torch_states.shape
    a_matrices = []
    for a in range(torch_actions.shape[0]):
        a_dist = distance(torch_actions[a][:, None],
                          torch_states[None, :],
                          dim=2)
        a_prob = F.softmax(-a_dist/tau, dim=1)
        a_matrices.append(a_prob.unsqueeze(0))
    transitions = torch.cat(a_matrices, dim=0)
    return transitions


def get_state_encoder(z_dim, env, num_actions):
    """
    Return encoder CNN with the same architecture, but different input sizes
    depending on the environment
    """
    activation = F.relu
    if env == "room":
        h = 48
        w = 48
        channels = 3
    elif env == "fashion_translations":
        h = 32
        w = 32
        channels = 1
    elif env == "cartpole":
        h = 4
        w = 1
        channels = 1

    if env == "cartpole":
        encoder = EncoderMLP(h, z_dim, activation=activation, mid=64,
                             mid2=32)
    else:
        encoder = EncoderCNN(channels, z_dim, h=h, w=w,
                             activation=activation)
    return encoder




def one_hot(action, n_actions):
    """
    Turn integer action into one-hot action
    """
    zeros = np.zeros((action.shape[0], n_actions))
    zeros[np.arange(action.shape[0]), action.cpu()] = 1
    return torch.FloatTensor(zeros).to(action.device)


def unsqueeze_samples(x, n):
    """
    """
    bn, d = x.shape
    x = x.reshape(bn//n, n, d)
    return x


def store_model(states, rewards, actions, folder):
    """
    Store states, actions, rewards
    """
    states.dump(os.path.join(folder, f"sampled_states.npy"))
    rewards.dump(os.path.join(folder, f"sampled_rewards.npy"))
    actions.dump(os.path.join(folder, f"sampled_actions.npy"))


def progress_bar(batch_idx, total, length=100, decimals=1, fill='█',
                prefix='progress', suffix='complete'):
    """
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (batch_idx / float(total)))
    filledLength = int(length * batch_idx // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
