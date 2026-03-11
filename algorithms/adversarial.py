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
            La profundidad disminuye cuando todos los agentes han jugado una vez (ply completo).
            """

            # Condición de parada: estado terminal o profundidad agotada
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()

            # Calculamos el siguiente agente y si completamos un ply completo
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            legal_actions = state.get_legal_actions(agent_index)

            # Si no hay acciones disponibles, evaluamos el estado actual
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

        # Desde la raíz evaluamos todas las acciones del dron y elegimos la mejor
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = None
        best_value = float('-inf')
        num_agents = state.get_num_agents()

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            # El siguiente agente es el primer cazador (agente 1), misma profundidad
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = minimax(successor, next_agent, next_depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action


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
            Igual que minimax pero con poda alfa-beta para descartar ramas inútiles.
            alpha: el mejor valor que MAX puede garantizar en el camino actual.
            beta:  el mejor valor que MIN puede garantizar en el camino actual.
            """

            # Condición de parada: estado terminal o profundidad agotada
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
                value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    alpha = max(alpha, value)
                    # Poda: si ya superamos lo mejor que MIN puede garantizar, cortamos
                    if value > beta:
                        break
                return value
            else:
                # Nodo MIN: el cazador minimiza
                value = float('inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    beta = min(beta, value)
                    # Poda: si ya bajamos de lo mejor que MAX puede garantizar, cortamos
                    if value < alpha:
                        break
                return value

        # Raíz: igual que minimax, elegimos la acción con mayor valor
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        num_agents = state.get_num_agents()

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = alphabeta(successor, next_agent, next_depth, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent. Drone nodes are MAX, hunter nodes are chance nodes.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax.
        Los cazadores no son MIN puros: calculan el valor esperado (promedio) de sus sucesores.
        """

        def expectimax(state, agent_index, depth):
            """
            Igual que minimax pero los nodos de los cazadores calculan el promedio,
            no el mínimo. Esto modela que el cazador actúa aleatoriamente a veces.
            """

            # Condición de parada
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            legal_actions = state.get_legal_actions(agent_index)

            if not legal_actions:
                return self.evaluation_function(state)

            if agent_index == 0:
                # Nodo MAX: el dron maximiza igual que antes
                best_value = float('-inf')
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = expectimax(successor, next_agent, next_depth)
                    best_value = max(best_value, value)
                return best_value
            else:
                # Nodo de azar: promedio de todos los sucesores (distribución uniforme)
                total = 0.0
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    total += expectimax(successor, next_agent, next_depth)
                return total / len(legal_actions)

        # Raíz: elegimos la acción del dron con mayor valor esperado
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = None
        best_value = float('-inf')
        num_agents = state.get_num_agents()

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            value = expectimax(successor, next_agent, next_depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
