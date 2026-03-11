from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from world.game_state import GameState


from algorithms.utils import bfs_distance, dijkstra

def evaluation_function(state) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """

    # Estados terminales: retornamos el valor máximo o mínimo directamente
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending = state.get_pending_deliveries()

    score = 0.0

    # --- Factor 1: distancia al punto de entrega más cercano ---
    # Usamos dijkstra para respetar el costo real del terreno.
    # Cuanto más cerca esté el dron de una entrega, mejor.
    if pending:
        min_delivery_cost = float('inf')
        for delivery in pending:
            cost, _ = dijkstra(layout, drone_pos, delivery)
            if cost < min_delivery_cost:
                min_delivery_cost = cost

        # Penalizamos por la distancia: más lejos = peor puntaje
        if min_delivery_cost == float('inf'):
            score -= 200
        else:
            score -= min_delivery_cost * 2

    # --- Factor 2: cantidad de entregas pendientes ---
    # Menos entregas pendientes es mejor (estamos más cerca de ganar).
    score -= len(pending) * 50

    # --- Factor 3: distancia de los cazadores al dron ---
    # Si el cazador está lejos, el dron está más seguro.
    # Usamos BFS con restricción de cazadores (solo caminan por terreno libre).
    for hunter_pos in hunter_positions:
        dist = bfs_distance(layout, hunter_pos, drone_pos, hunter_restricted=True)
        if dist == float('inf'):
            # El cazador no puede alcanzarnos: ninguna penalización
            continue
        elif dist <= 1:
            # El cazador está adyacente: situación muy peligrosa
            score -= 500
        elif dist <= 3:
            # El cazador está cerca: penalizamos bastante
            score -= 150
        else:
            # El cazador está lejos: pequeña recompensa por la distancia
            score += dist * 5

    return max(-999.0, min(999.0, score))
