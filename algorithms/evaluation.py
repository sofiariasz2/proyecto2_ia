from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.game_state import GameState

from algorithms.utils import bfs_distance, dijkstra


def evaluation_function(state) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    Diseño:
    (a) Distancia real (dijkstra) al punto de entrega más cercano: gradiente continuo
        que siempre empuja al dron hacia adelante.
    (b) Peligro de cada cazador: función 1/dist^2 para un gradiente suave y continuo
        que evita mesetas (el problema de los umbrales discretos).
    (c) Urgencia de entrega: si el dron puede llegar a una entrega antes que cualquier
        cazador, se recompensa fuertemente para que el dron se comprometa.
    (d) Score del juego como desempate: premia acumular puntos reales.
    """

    # Estados terminales
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending = state.get_pending_deliveries()

    score = 0.0

    # --- Factor 1: progreso hacia la entrega más cercana ---
    # Usamos dijkstra para respetar el costo real del terreno.
    # El peso es alto (x10) para que el dron siempre tenga un incentivo claro de avanzar.
    if pending:
        min_delivery_cost = float('inf')
        nearest_delivery = None
        for delivery in pending:
            cost, _ = dijkstra(layout, drone_pos, delivery)
            if cost < min_delivery_cost:
                min_delivery_cost = cost
                nearest_delivery = delivery

        if min_delivery_cost == float('inf'):
            # No hay ruta posible: penalización fuerte
            score -= 300
        else:
            score -= min_delivery_cost * 10

        # --- Factor 3: urgencia ---
        # Si el dron llega a la entrega antes que todos los cazadores,
        # recompensamos fuertemente para que se comprometa con esa entrega.
        if nearest_delivery is not None and min_delivery_cost < float('inf'):
            drone_arrives = min_delivery_cost
            min_hunter_to_delivery = float('inf')
            for hunter_pos in hunter_positions:
                h_dist = bfs_distance(layout, hunter_pos, nearest_delivery, hunter_restricted=True)
                if h_dist < min_hunter_to_delivery:
                    min_hunter_to_delivery = h_dist

            if drone_arrives < min_hunter_to_delivery:
                # El dron gana la carrera: recompensa proporcional a la ventaja
                advantage = min_hunter_to_delivery - drone_arrives
                score += advantage * 15

    # --- Factor 2: penalización por proximidad de cazadores ---
    # Usamos 1/dist^2 en vez de umbrales discretos para tener un gradiente continuo.
    # Esto evita las "mesetas" donde múltiples estados tienen el mismo puntaje
    # y el dron oscila sin dirección.
    for hunter_pos in hunter_positions:
        dist = bfs_distance(layout, hunter_pos, drone_pos, hunter_restricted=True)

        if dist == float('inf'):
            # El cazador no puede alcanzarnos por su terreno restringido
            continue
        elif dist == 0:
            # Captura inminente (no debería pasar antes de is_lose, pero por si acaso)
            score -= 900
        else:
            # Penalización suave: cuanto más cerca, mucho más peligroso
            # dist=1 → -400, dist=2 → -100, dist=3 → -44, dist=5 → -16, dist=10 → -4
            score -= 400.0 / (dist ** 2)

    # --- Factor 4: score real del juego como desempate ---
    # Premia haber completado entregas anteriores y evita que dos estados
    # con igual distancia tengan exactamente el mismo puntaje.
    score += state.get_score() * 0.5

    return max(-999.0, min(999.0, score))