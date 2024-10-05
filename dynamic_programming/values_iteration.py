import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for _ in range(max_iter):
        new_values = np.zeros(mdp.observation_space.n)
        for s in range(mdp.observation_space.n):
            q_values = []
            for a in range(mdp.action_space.n):
                next_state, reward, _ = mdp.P[s][a]
                q_values.append(reward + gamma * values[next_state])
            new_values[s] = max(q_values)
        
        if np.allclose(values, new_values):
            break
        
        values = new_values
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        delta = 0
        new_values = np.copy(values)
        for i in range(env.height):
            for j in range(env.width):
                if env.grid[i, j] == 'W':  # Ignorer les murs
                    continue
                # Définir l'état courant
                env.set_state(i, j)
                q_values = []
                for action in range(env.action_space.n):
                    next_state, reward, is_done, _ = env.step(action, make_move=False)
                    ni, nj = next_state
                    if is_done:
                        q = reward
                    else:
                        q = reward + gamma * values[ni][nj]
                    q_values.append(q)
                if q_values:
                    new_v = max(q_values)
                else:
                    new_v = 0
                
                delta = max(delta, abs(values[i, j] - new_v))
                new_values[i, j] = new_v
        values = new_values
        if delta < theta:
            break
    # END SOLUTION
    return values


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        delta = 0
        new_values = np.copy(values)
        
        for i in range(env.height):
            for j in range(env.width):
                if env.grid[i, j] == 'W':  # Ignorer les murs
                    continue
                
                env.set_state(i, j)
                q_values = []
                
                for action in range(env.action_space.n):
                    next_states = env.get_next_states(action)
                    expected_value = 0
                    
                    for next_state, reward, prob, is_done, _ in next_states:
                        ni, nj = next_state
                        if is_done:
                            expected_value += prob * reward
                        else:
                            expected_value += prob * (reward + gamma * values[ni][nj])
                    
                    q_values.append(expected_value)
                
                if q_values:
                    new_v = max(q_values)
                else:
                    new_v = 0
                
                delta = max(delta, abs(values[i, j] - new_v))
                new_values[i, j] = new_v
        
        values = new_values
        
        if delta < theta:
            break
    
    return values
    # END SOLUTION
