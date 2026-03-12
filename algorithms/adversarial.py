from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """

        def minimax(state, agent_index, depth):
            """
            Función recursiva del minimax.
            agent_index=0 es el dron (MAX), cualquier otro es un cazador (MIN).
            La profundidad se decrementa solo cuando volvemos al dron (ply completo).
            """

            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            # La profundidad baja solo cuando terminamos un ply completo (volvemos al dron)
            next_depth = depth - 1 if next_agent == 0 else depth

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            if agent_index == 0:
                # Nodo MAX: el dron busca maximizar
                best_value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = minimax(successor, next_agent, next_depth)
                    best_value = max(best_value, value)
                return best_value
            else:
                # Nodo MIN: el cazador busca minimizar
                best_value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = minimax(successor, next_agent, next_depth)
                    best_value = min(best_value, value)
                return best_value

        # Raíz: evaluamos cada acción del dron y elegimos la de mayor valor.
        # Si hay empate, elegimos aleatoriamente entre los mejores para romper ciclos.
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        scored_actions = []

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = minimax(successor, next_agent, next_depth)
            scored_actions.append((value, action))

        # Elegimos aleatoriamente entre todas las acciones con el valor máximo.
        # Esto rompe empates y evita que el dron oscile entre dos posiciones iguales.
        best_value = max(v for v, _ in scored_actions)
        best_actions = [a for v, a in scored_actions if v == best_value]
        return random.choice(best_actions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """

        def alphabeta(state, agent_index, depth, alpha, beta):
            """
            Minimax con poda alfa-beta.
            alpha: mejor garantía de MAX en el camino actual.
            beta:  mejor garantía de MIN en el camino actual.
            """

            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            if agent_index == 0:
                # Nodo MAX
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    alpha = max(alpha, value)
                    # Poda estricta: cortamos si superamos lo mejor de MIN
                    if value > beta:
                        break
                return value
            else:
                # Nodo MIN
                value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    beta = min(beta, value)
                    # Poda estricta: cortamos si bajamos de lo mejor de MAX
                    if value < alpha:
                        break
                return value

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        alpha = float('-inf')
        beta = float('inf')
        scored_actions = []

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = alphabeta(successor, next_agent, next_depth, alpha, beta)
            scored_actions.append((value, action))
            alpha = max(alpha, value)

        # Misma lógica de desempate aleatorio que Minimax
        best_value = max(v for v, _ in scored_actions)
        best_actions = [a for v, a in scored_actions if v == best_value]
        return random.choice(best_actions)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent. Drone nodes are MAX, hunter nodes are chance nodes.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax.
        Los cazadores calculan el valor esperado (promedio) en vez del mínimo.
        """

        def expectimax(state, agent_index, depth):
            """
            Igual que minimax pero los nodos de los cazadores usan una política mixta:
            con probabilidad self.prob actúan al azar (promedio uniforme),
            con probabilidad (1 - self.prob) actúan greedy (minimizan el valor).
            """

            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            if agent_index == 0:
                # Nodo MAX: el dron maximiza
                best_value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = expectimax(successor, next_agent, next_depth)
                    best_value = max(best_value, value)
                return best_value
            else:
                # Nodo de azar: política mixta greedy/aleatoria
                values = [
                    expectimax(state.generate_successor(agent_index, action), next_agent, next_depth)
                    for action in legal_actions
                ]
                avg_value = sum(values) / len(values)   # cazador actúa al azar
                min_value = min(values)                  # cazador actúa greedy (minimiza)
                return self.prob * avg_value + (1.0 - self.prob) * min_value

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        scored_actions = []

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = expectimax(successor, next_agent, next_depth)
            scored_actions.append((value, action))

        # Desempate aleatorio igual que los otros agentes
        best_value = max(v for v, _ in scored_actions)
        best_actions = [a for v, a in scored_actions if v == best_value]
        return random.choice(best_actions)