import numpy as np


class DynamicProgramming:
    def __init__(self, MDP):
        self.R = MDP.R  # |A|x|S|
        self.P = MDP.P  # |A|x|S|x|S|
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    ####Helpers####
    def extractRpi(self, pi):
        '''
        Returns R(s, pi(s)) for all states. Thus, the output will be an array of |S| entries. 
        This should be used in policy evaluation and policy iteration. 

        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers, each of which specifies an action (row) for a given state s.

        HINT: Given an m x n matrix A, the expression

        A[row_indices, col_indices] (len(row_indices) == len(col_indices))

        returns a matrix of size len(row_indices) that contains the elements

        A[row_indices[i], col_indices[i]] in a row for all indices i.
        '''
        return self.R[pi, np.arange(len(self.R[0]))]

    def extractPpi(self, pi):
        '''
        Returns P^pi: This is a |S|x|S| matrix where the (i,j) entry corresponds to 
        P(j|i, pi(i))


        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers
        '''
        return self.P[pi, np.arange(len(self.P[0]))]

    ####Value Iteration###
    def computeVfromQ(self, Q, pi):
        '''
        Returns the V function for a given Q function corresponding to a deterministic policy pi. Remember that

        V^pi(s) = Q^pi(s, pi(s))

        Parameter Q: Q function
        Precondition: An array of |S|x|A| numbers

        Parameter pi: Policy
        Precondition: An array of |S| integers
        '''
        # Iterate over the number of states and evaluate Q(s, pi(s))
        return np.array([Q[s, pi[s]] for s in range(self.nStates)])


    def computeQfromV(self, V):
        '''
        Returns the Q function given a V function corresponding to a policy pi. The output is an |S|x|A| array.

        Use the bellman equation for Q-function to compute Q from V.

        Parameter V: value function
        Precondition: An array of |S| numbers
        '''
        # Initialize Q to all zeros
        Q = np.zeros((self.nStates, self.nActions))
        # Iterate over states and actions
        for s in range(self.nStates):
            for a in range(self.nActions):
                # Add the immediate reward
                q_sa = self.R[a, s]
                # Expected later reward, discounted by gamma
                for sp in range(self.nStates):
                    q_sa += self.discount * self.P[a, s, sp] * V[sp]
                Q[s, a] = q_sa
        return Q


    def extractMaxPifromQ(self, Q):
        '''
        Returns the policy pi corresponding to the Q-function determined by 

        pi(s) = argmax_a Q(s,a)


        Parameter Q: Q function 
        Precondition: An array of |S|x|A| numbers
        '''
        # For every state, return the action a that maximizes Q[s, a]
        return np.array([np.argmax(Q[s, :]) for s in range(self.nStates)])

    def extractMaxPifromV(self, V):
        '''
        Returns the policy corresponding to the V-function. Compute the Q-function
        from the given V-function and then extract the policy following

        pi(s) = argmax_a Q(s,a)

        Parameter V: V function 
        Precondition: An array of |S| numbers
        '''
        # Compute Q from V and pass this to the previous function to compute pi from Q
        return self.extractMaxPifromQ(self.computeQfromV(V))


    def valueIterationStep(self, Q):
        '''
        Returns the Q function after one step of value iteration. The input
        Q can be thought of as the Q-value at iteration t. Return Q^{t+1}.

        Parameter Q: value function 
        Precondition: An array of |S|x|A| numbers
        '''
        # Compute the best policy from the given Q
        maxpi = self.extractMaxPifromQ(Q)
        # Updated Q function is the Q corresponding to the best policy
        return self.computeQfromV(self.computeVfromQ(Q, maxpi))


    def valueIteration(self, initialQ, tolerance=0.01):
        '''
        This function runs value iteration on the input initial Q-function until 
        a certain tolerance is met. Specifically, value iteration should continue to run until 
        ||Q^t-Q^{t+1}||_inf <= tolerance. Recall that for a vector v, ||v||_inf is the maximum 
        absolute element of v. 

        This function should return the policy, value function, number
        of iterations required for convergence, and the end epsilon where the epsilon is 
        ||Q^t-Q^{t+1}||_inf. 

        Parameter initialQ:  Initial Q-function
        Precondition: array of |S|x|A| entries

        Parameter tolerance: threshold threshold on ||Q^t-Q^{t+1}||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        # Current Q-function is the initial one
        iterId = 0
        Q_curr = initialQ
        # Iterate until convergence
        while True:
            iterId += 1
            # Compute new Q-function and check for convergence
            Q_new = self.valueIterationStep(Q_curr)
            epsilon = np.max(np.abs(Q_new - Q_curr))
            # Break if converged
            if epsilon <= tolerance:
                break
            # Update current Q-function and go to next step if not converged
            Q_curr = Q_new
        maxpi = self.extractMaxPifromQ(Q_new)
        return maxpi, self.computeVfromQ(Q_new, maxpi), iterId, epsilon


    ### EXACT POLICY EVALUATION  ###
    def exactPolicyEvaluation(self, pi):
        '''
        Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma P^pi V^pi

        Return the value function

        Parameter pi: Deterministic policy 
        Precondition: array of |S| integers

        '''
        # Solve the system AV = B, where A = (1 - gamma * Ppi) and B = Rpi
        A = np.identity(self.nStates) - self.discount * self.extractPpi(pi)
        B = self.extractRpi(pi)
        return np.linalg.solve(A, B)


    ### APPROXIMATE POLICY EVALUATION ###
    def approxPolicyEvaluation(self, pi, tolerance=0.01):
        '''
        Evaluate a policy using approximate policy evaluation. Like value iteration, approximate 
        policy evaluation should continue until ||V_n - V_{n+1}||_inf <= tolerance. 

        Return the value function, number of iterations required to get to exactness criterion, and final epsilon value.

        Parameter pi: Deterministic policy 
        Precondition: array of |S| integers

        Parameter tolerance: threshold threshold on ||V^n-V^n+1||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        n_iters = 0
        # Initialize value function to all zeros and extract Rpi, Ppi
        V_curr = np.zeros(self.nStates)
        Rpi, Ppi = self.extractRpi(pi), self.extractPpi(pi)
        # Iterate until convergence
        while True:
            n_iters += 1
            # Apply formula to current V and check for convergence
            V_new = Rpi + self.discount * Ppi @ V_curr
            epsilon = np.max(np.abs(V_new - V_curr))
            # Break if converged
            if epsilon <= tolerance:
                break
            # Update current V and go to next step if not converged
            V_curr = V_new
        return V_new, n_iters, epsilon

    def policyIterationStep(self, pi, exact):
        '''
        This function runs one step of policy iteration, followed by one step of policy improvement. Return
        pi^{t+1} as a new numpy array. Do not modify pi^t.

        Parameter pi: Current policy pi^t
        Precondition: array of |S| integers

        Parameter exact: Indicate whether to use exact policy evaluation 
        Precondition: boolean
        '''
        # Find the value of the current policy, exactly or approximately
        if exact:
            V = self.exactPolicyEvaluation(pi)
        else:
            V = self.approxPolicyEvaluation(pi)[0]
        # Compute the Q function corresponding to the calculated V and extract the best policy
        return self.extractMaxPifromQ(self.computeQfromV(V))

    def policyIteration(self, initial_pi, exact):
        '''

        Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a Q^pi(s,a)).


        This function should run policyIteration until convergence where convergence 
        is defined as pi^{t+1}(s) == pi^t(s) for all states s.

        Return the final policy, value-function for that policy, and number of iterations
        required until convergence.

        Parameter initial_pi:  Initial policy
        Precondition: array of |S| entries

        Parameter exact: Indicate whether to use exact policy evaluation 
        Precondition: boolean

        '''
        iterId = 0
        # Set current policy to be the given initial one one
        pi_curr = initial_pi
        # Iterate until convergence
        while True:
            iterId += 1
            # Update policy and break if converged
            pi_new = self.policyIterationStep(pi_curr, exact)
            if np.array_equal(pi_new, pi_curr):
                break
            # Proceed to next step if not converged
            pi_curr = pi_new
        # Return best policy and value function, calculated exactly or approx.
        if exact:
            return pi_new, self.exactPolicyEvaluation(pi_new), iterId
        return pi_new, self.approxPolicyEvaluation(pi_new)[0], iterId
