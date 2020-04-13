import gym
from gym import spaces
from gym.utils import seeding

# Create an environment for the agent to act in. The class Artificial_Economy defines the parameters of action (range of
# admissible values for i and π), stores the hidden values of parameters A and B, and computes rewards and next
# step inflation based on the agent's action.

#Two modes are provided. In this first, the action space (raising and lowering the interest rate) are discretized.
#The banker has 7 choices. First, he can do nothing. The next 6 choices are to do a [large, medium, or small]
#[increase or decrease] of the interest rate. This allows for implementation of a Q-Learning solution 
#to the problem. A second mode allows the banker to choose any continuous value for the interst rate, 
#in which case a more sophisticated solution is required.

class Artificial_Economy(gym.Env):
    
    def __init__(self, A=1.1, B=-0.1, α=1, mode='discrete'):
        self.A = A
        self.B = B
        self.α = α
        self.π_0 = np.random.uniform(-1, 10)
        self.i_0 = np.random.uniform(-2, 20)

        self._max_episode_steps = None
        self._current_step = 0
        self.mode = mode
        
        self.goal_π = 0 #The goal paramter defines when the simulation is "over". Here, the inflation target is 0.
        
        self.min_π = -1.
        self.max_π = 10.
        self.min_i = -2.
        self.max_i = 20.
        
        self.low = np.array([self.min_π, self.min_i], dtype=np.float64)
        self.high = np.array([self.max_π, self.max_i], dtype=np.float64)
        
        
        if self.mode == 'discrete':
            self.action_space = [-5, -3, -1, 0, 1, 3, 5]
            self.force = 0.01
        elif self.mode == 'continuous':
            self.action_space = spaces.Box(low=self.min_i, high=self.max_i, shape=(1, ), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=self.min_π, high=self.max_π, shape=(1,), dtype=np.float64)
        
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        self.π, self.i = self.state
         
        if self.mode == 'discrete':
            self.i += (action-1) * self.force
            self.i = min(self.i, self.max_i)
            self.i = max(self.i, self.min_i)
        else: 
            self.i = action
        
        self.π = self.A * self.π + self.B * self.i
        
        if self._max_episode_steps is not None:
            if self._current_step <= self._max_episode_steps:
                done = bool(np.abs(self.π - self.goal_π) < 0.1)
                self._current_step += 1
            else:
                done = True

        else:
            done = bool(np.abs(self.π - self.goal_π) < 0.1)

        reward = -(self.A**2 * self.π**2 + 2 * self.A * self.B * self.π * self.i + self.B**2 * self.i ** 2 + self.α * self.i **2)

        self.state = (self.π, self.i)
        
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=self.min_π, high=self.max_π), 
                               self.np_random.uniform(low=self.min_i, high=self.max_i)])
        self._current_step = 0
        return self.state
    
    def get_inflation(self):
        if isinstance(self.π, np.ndarray):
            inf = self.π.squeeze().item()
        else:
            inf = self.π
        return inf
    
    def get_interest(self):
        if isinstance(self.i, np.ndarray):
            intr = self.i.squeeze().item()
        else:
            intr = self.i
        return intr