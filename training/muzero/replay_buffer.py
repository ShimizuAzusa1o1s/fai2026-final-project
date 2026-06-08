import numpy as np
import random

class ReplayBuffer:
    """
    Stores full game trajectories for MuZero unrolled training.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, trajectory):
        """
        trajectory: dict with keys:
            'obs': list of np.arrays [T, 230]
            'actions': list of ints [T]
            'rewards': list of floats [T]
            'policies': list of dicts [T] mapping action -> prob
            'values': list of floats [T]
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, unroll_steps=5):
        """
        Samples a batch of unrolled sequences.
        Returns:
            obs_batch: list of np.arrays (initial state)
            actions_batch: list of lists of actions (length unroll_steps)
            rewards_batch: list of lists of rewards (length unroll_steps)
            policies_batch: list of lists of policy arrays (length unroll_steps)
            values_batch: list of lists of values (length unroll_steps)
        """
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        policies_batch = []
        values_batch = []
        
        for _ in range(batch_size):
            traj = random.choice(self.buffer)
            T = len(traj['obs'])
            
            # Start step
            start_step = random.randint(0, T - 1)
            
            obs_batch.append(traj['obs'][start_step])
            
            actions = []
            rewards = []
            policies = []
            values = []
            
            for i in range(unroll_steps):
                step = start_step + i
                if step < T:
                    actions.append(traj['actions'][step])
                    rewards.append(traj['rewards'][step])
                    
                    policy_dict = traj['policies'][step]
                    policy_array = np.zeros(105, dtype=np.float32)
                    for a, p in policy_dict.items():
                        if a < 105:
                            policy_array[a] = p
                    policies.append(policy_array)
                    
                    values.append(traj['values'][step])
                else:
                    # Pad sequence if we roll past the end of the game
                    actions.append(0) # dummy action
                    rewards.append(0.0)
                    policies.append(np.zeros(105, dtype=np.float32))
                    values.append(0.0)
                    
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            policies_batch.append(policies)
            values_batch.append(values)
            
        return obs_batch, actions_batch, rewards_batch, policies_batch, values_batch
        
    def __len__(self):
        return len(self.buffer)
