import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    """
    samples N trajectories using the current policy
    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []
    for _ in range(N):
        steps = 0
        grads, rewards = [], []
        obs = env.reset()
        while steps < 200:
            phis = utils.extract_features(obs, 2)
            # Compute probability for each action according to the policy
            probs = utils.compute_action_distribution(theta, phis)
            if np.random.uniform() < probs[0][0]:
                action = 0
            else:
                action = 1
            # Compute gradients and rewards
            grads.append(utils.compute_log_softmax_grad(theta, phis, action))
            # Take the action and proceed to next step
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1
            # Check if done
            if done:
                break
        total_grads.append(grads)
        total_rewards.append(rewards)
    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """
    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100, 1)
    env = gym.make('CartPole-v0')
    env.seed(12345)
    episode_rewards = []

    for _ in range(T):
        grads, rewards = sample(theta, env, N)
        fisher = utils.compute_fisher_matrix(grads, lamb=lamb)
        v_grad = utils.compute_value_gradient(grads, rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)
        theta += eta * np.linalg.inv(fisher) @ v_grad
        episode_rewards.append(np.mean([np.sum(r) for r in rewards]))
        env.reset()
    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.savefig("./Answers.pdf", bbox_inches="tight")
    plt.show()
