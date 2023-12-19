from learner import *
from dataset import *
from common import DISCRETE_ENVS, plot_losses
import torch
import gym
import argparse
from tqdm import tqdm
import os
from dataset import ExpertData, ExpertDataset

def experiment(args):
    # expert dataset loading
    save_path = os.path.join(args.data_dir, args.env+'_dataset.pt')
    # Initialize empty dataset for DAGGER if flag is False
    if args.dagger and not args.dagger_dataset:
        expert_dataset = ExpertDataset(ExpertData(torch.tensor([]), torch.tensor([], dtype=int)))
    # Otherwise, load dataset in filepath given
    else:
        expert_dataset = torch.load(save_path)
    # Create env
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # policy initialization
    if not args.dagger:
        learner = BC(state_dim, action_dim, args)
    else:
        learner = DAGGER(state_dim, action_dim, args)

    if args.dagger:
        epoch_losses = []
        # Keep track of losses at each epoch for plotting
        dagger_losses = []
        for _ in tqdm(range(1, args.dagger_epochs + 1)):
            # Rollout new data
            # Iterate over the number of batches
            for _ in range(args.batch_size):
                # Add states and expert actions to ExpertDataset class
                expert_dataset.add_data(learner.rollout(env, args.num_rollout_steps))
                env.reset()
            # Create dataloader object
            dataloader = get_dataloader(expert_dataset, args)
            # Supervised learning step
            supervision_loss = []
            for _ in tqdm(range(1, args.dagger_supervision_steps + 1)):
                loss = 0.0
                # Gradient descent
                for batch in dataloader:
                    loss += learner.learn(batch[0], batch[1])
                supervision_loss.append(loss)
                dagger_losses.append(loss / len(dataloader))
            epoch_losses.append(np.mean(supervision_loss))

    else:
        epoch_losses = []
        dataloader = get_dataloader(expert_dataset, args)
        for _ in tqdm(range(1, args.bc_epochs + 1)):
            loss = 0.0
            # Learning occurs over all batches in the dataloader
            # Start by 'for batch in dataloader:' ... 
            for batch in dataloader:
                loss += learner.learn(batch[0], batch[1])
            epoch_losses.append(loss / len(dataloader))
    
    # Plotting
    epochs = np.arange(1, args.dagger_epochs * args.dagger_supervision_steps + 1 if args.dagger else args.bc_epochs + 1)
    plot_losses(epochs, dagger_losses if args.dagger else epoch_losses, args.env, args.dagger)
            
    # saving policy
    if not os.path.exists(args.policy_save_dir):
        bc_path = os.path.join(args.policy_save_dir, 'bc')
        dagger_path = os.path.join(args.policy_save_dir, 'dagger')
        os.makedirs(bc_path)
        os.makedirs(dagger_path)
    
    if args.dagger:
        policy_save_path = os.path.join(args.policy_save_dir, 'dagger' ,f'{args.env}.pt')
    else:
        policy_save_path = os.path.join(args.policy_save_dir, 'bc', f'{args.env}.pt')
    
    learner.save(policy_save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Imitation learning')

    # general + env args
    parser.add_argument('--data_dir', default='./data', help='dataset directory')
    parser.add_argument('--env', default='CartPole-v0', help='environment')
    
    # learning args
    parser.add_argument('--bc_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_dataset_samples', type=int, default=10000, help='number of samples to start dataset off with')
    
    # DAGGER args
    parser.add_argument('--dagger', action='store_true', help='flag to run DAGGER')
    parser.add_argument('--expert_save_path', default='./expert_policies')
    parser.add_argument('--num_rollout_steps', type=int, default=100, help='number of steps to roll out with the policy')
    parser.add_argument('--dagger_epochs', type=int, default=10, help='number of steps to run dagger')
    parser.add_argument('--dagger_supervision_steps', type=int, default=10, help='number of epochs for supervised learning step within dagger')
    
    # model saving args
    parser.add_argument('--policy_save_dir', default='./learned_policies', help='policy saving directory')

    # Argument to initialize DAGGER to something other than empty
    parser.add_argument('--dagger_dataset', action='store_true', help='flag to not use empty initial dataset')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    experiment(args)