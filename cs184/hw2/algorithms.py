from env_MAB import *
from functools import lru_cache


def random_argmax(a):
    '''
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    '''
    return np.random.choice(np.where(a == a.max())[0])


# Helper function to get arm with highest number of successes
def get_best_arm(record):
    # Return the argmax of the empirical means
    return random_argmax(np.divide(record[:, 1], record.sum(axis=1)))


class Explore():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        # Get no. of pulls for each arm by adding successes + failures
        pulls = self.MAB.get_record().sum(axis=1)
        # Pull the arm with the least pulls
        self.MAB.pull(random_argmax(-1 * pulls))


class Greedy():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        K = self.MAB.get_K()
        record = self.MAB.get_record()
        t = int(record.sum())
        # First pull every arm exactly once
        if t < K:
            self.MAB.pull(t)
        # If we have pulled every arm, find best one and pull it
        elif t == K:
            self.best_arm = get_best_arm(record)
            self.MAB.pull(self.best_arm)
        # Continue exploiting best arm otherwise
        else:
            self.MAB.pull(self.best_arm)


class ETC():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        # Find Ne
        T, K = self.MAB.get_T(), self.MAB.get_K()
        self.Ne = int(np.floor((T * np.sqrt(np.log(2 * K / delta) / 2) / K)**(2/3)))
        # Boolean to store whether we are exploiting or not
        self.exploit = False

    def reset(self):
        self.MAB.reset()
        # Reset exploitation phase to false
        self.exploit = False

    def play_one_step(self):
        K = self.MAB.get_K()
        record = self.MAB.get_record()
        # If we are exploiting, pull the best arm
        if self.exploit:
            self.MAB.pull(self.best_arm)
        # If we are exploring, find arms we have not pulled Ne times
        else:
            not_explored = np.where(record.sum(axis=1) < self.Ne)[0]
            # Choose random arm we have not fully explored if there are any
            if not_explored.shape[0] > 0:
                self.MAB.pull(np.random.choice(not_explored))
            # If we have explored each arm Ne times, find best arm and exploit
            else:
                self.best_arm = get_best_arm(record)
                self.MAB.pull(self.best_arm)
                self.exploit = True

class Epgreedy():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()
    
    def play_one_step(self):
        K = self.MAB.get_K()
        # Get current time step t
        record = self.MAB.get_record()
        t = int(record.sum())
        # First pull each arm once
        if t < K:
            self.MAB.pull(t)
        # Otherwise, explore with probability epsilon_t, exploit else
        else:
            # Calculate epsilon_t
            eps_t = (K * np.log(t) / t)**(1/3)
            # Explore with probability epsilon_t
            if np.random.uniform() < eps_t:
                self.MAB.pull(np.random.choice(K))
            # Pull best arm with probability 1 - epsilon_t
            else:
                self.MAB.pull(get_best_arm(record))


class UCB():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        K, T = self.MAB.get_K(), self.MAB.get_T()
        # Find UCB of every arm
        denom = record.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide-by-zero warnings
            # First find the mu^hat
            UCB_arr = np.divide(record[:, 1], denom)
            # Add the upper bound
            UCB_arr += np.sqrt(np.divide(np.log(K * T / self.delta), denom))
            # Set elements where we divided by zero to infinity
            UCB_arr[~np.isfinite(UCB_arr)] = np.inf
        # Pull the arm with the highest UCB
        self.MAB.pull(random_argmax(UCB_arr))


class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
 
    def play_one_step(self):
        '''
        Implement one step of the Thompson sampling algorithm. 
        '''
        K = self.MAB.get_K()
        record = self.MAB.get_record()
        mu_draws = np.zeros(K)
        # Draw from the corresponding Beta distribution for each arm
        for arm in range(K):
            # alpha = 1 + no. successes, beta = 1 + no. failures
            mu_draws[arm] = np.random.beta(1 + record[arm][1], 1 + record[arm][0])
        # Pull arm with highest mu draw
        self.MAB.pull(random_argmax(mu_draws))
