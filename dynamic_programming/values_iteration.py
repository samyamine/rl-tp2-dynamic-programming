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
    for iteration in range(max_iter):
        updated_values = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                highest_value = float("-inf")

                if env.grid[i, j] in ["P", "N", "W"]:
                    continue

                for act in range(env.action_space.n):
                    env.set_state(i, j)

                    nxt_state, gain, terminal, _ = env.step(act, False)

                    if not terminal:
                        result = gain + gamma * values[nxt_state[0]][nxt_state[1]]
                    else:
                        result = gain

                    if result > highest_value:
                        highest_value = result

                updated_values[i][j] = highest_value

        max_change = np.max(np.abs(updated_values - values))

        if max_change < theta:
            break
        values = updated_values.copy()

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
    diff = float("inf")
    iteration_count = 0

    while iteration_count < max_iter:
        temp_values = values.copy()

        for x in range(4):
            for y in range(4):
                env.set_state(x, y)
                diff = value_iteration_per_state(env, values, gamma, temp_values, diff)
        
        if diff < theta:
            break
        
        iteration_count += 1
    # END SOLUTION
    return values
