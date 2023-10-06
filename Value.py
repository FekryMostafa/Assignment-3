import numpy as np

P = np.array([
    [0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],  # A
    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # B
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.2],  # C
    [0.0, 0.0, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # D
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # E
    [0.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],  # F
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.6, 0.0],  # G
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9],  # H
    [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],  # I
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # J
])

states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


gamma = 0.9
R = np.array([-1, -1, -1, -1, 2, -1, 2, 2, -1, 15]).reshape(-1, 1)
I = np.identity(len(states))
V = np.matmul(np.linalg.inv(I-(gamma*P)), R)

# Print the Value Function V
print("Value Function V for each state:")
for state, value in zip(states, V):
    print(f"{state}: {value[0]}")

#The MDP is solved
#The optimal policy is the action that takes the agent from one state to another state with the maximum value function.
#The max reward is 15.31149 A to H to J 

#State A -> State H
#State B -> State D
#State C -> State J
#State D -> State E
#State E -> State I
#State F -> State E
#State G -> State E
#State H -> State J
#State I -> State H
#State J -> None