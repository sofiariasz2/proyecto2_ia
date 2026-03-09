from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking basico sin optimizaciones.
    Pruebo asignar drones a entregas uno por uno, y si una asignacion
    no cumple las restricciones, me devuelvo e intento con otro dron.
    """
    assignment: dict[str, str] = {}

    def backtrack() -> bool:
        # Si ya asigne todas las entregas, encontre solucion
        if csp.is_complete(assignment):
            return True

        # Tomo la primera entrega que no tenga dron asignado
        var = csp.get_unassigned_variables(assignment)[0]

        # Pruebo cada dron disponible para esta entrega
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                # Si el dron cumple las restricciones, lo asigno
                csp.assign(var, value, assignment)
                # Intento asignar las demas entregas
                if backtrack():
                    return True
                # Si no funciono, deshago y pruebo otro dron
                csp.unassign(var, assignment)

        # Ningun dron funciono para esta entrega, me devuelvo
        return False

    if backtrack():
        return assignment
    return None


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking con Forward Checking.
    Despues de cada asignacion, reviso las entregas vecinas y elimino
    drones que ya no serian validos. Si alguna entrega se queda sin
    opciones, me devuelvo inmediatamente sin seguir buscando.
    """
    assignment: dict[str, str] = {}

    def forward_check(var: str, value: str) -> tuple[dict[str, list[str]], bool]:
        """
        Reviso las entregas vecinas y elimino drones que ya no son validos.
        Retorno los valores eliminados (para poder restaurarlos) y si fue exitoso.
        """
        removals: dict[str, list[str]] = {}
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue
            removed: list[str] = []
            # Reviso cada dron posible del vecino
            for val in csp.domains[neighbor][:]:
                if not csp.is_consistent(neighbor, val, assignment):
                    # Este dron ya no es valido, lo quito
                    csp.domains[neighbor].remove(val)
                    removed.append(val)
            if removed:
                removals[neighbor] = removed
            # Si una entrega se quedo sin drones posibles, fallo de una vez
            if not csp.domains[neighbor]:
                return removals, False
        return removals, True

    def restore_domains(removals: dict[str, list[str]]) -> None:
        """Restauro los valores que elimine al hacer forward checking."""
        for neighbor, values in removals.items():
            csp.domains[neighbor].extend(values)

    def backtrack() -> bool:
        if csp.is_complete(assignment):
            return True

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                # Propago restricciones a los vecinos
                removals, success = forward_check(var, value)
                if success:
                    if backtrack():
                        return True
                # Restauro los dominios que modifique
                restore_domains(removals)
                csp.unassign(var, assignment)

        return False

    if backtrack():
        return assignment
    return None


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking con consistencia de arco (AC-3).
    Similar a Forward Checking, pero voy mas alla: si al eliminar opciones
    de un vecino se reducen sus posibilidades, tambien reviso los vecinos
    de ese vecino, creando una cadena de propagacion que descarta mas
    opciones invalidas antes de seguir buscando.
    """
    assignment: dict[str, str] = {}

    def ac3_propagate(var: str) -> tuple[dict[str, list[str]], bool]:
        """
        Despues de asignar una variable, elimino opciones invalidas
        de los vecinos y propago en cadena si es necesario.
        Retorno los valores eliminados y si fue exitoso.
        """
        removals: dict[str, list[str]] = {}
        # Cola de entregas que debo revisar
        queue = [n for n in csp.get_neighbors(var) if n not in assignment]

        while queue:
            xi = queue.pop(0)
            if xi in assignment:
                continue
            removed: list[str] = []
            for val in csp.domains[xi][:]:
                if not csp.is_consistent(xi, val, assignment):
                    csp.domains[xi].remove(val)
                    removed.append(val)
            if removed:
                # Guardo lo que elimine para poder restaurar despues
                if xi in removals:
                    removals[xi].extend(removed)
                else:
                    removals[xi] = removed
                # Si se quedo sin opciones, esta asignacion no sirve
                if not csp.domains[xi]:
                    return removals, False
                # Como este vecino cambio, reviso tambien sus vecinos
                for xk in csp.get_neighbors(xi):
                    if xk not in assignment and xk not in queue:
                        queue.append(xk)

        return removals, True

    def restore_domains(removals: dict[str, list[str]]) -> None:
        """Restauro los valores eliminados durante la propagacion."""
        for neighbor, values in removals.items():
            csp.domains[neighbor].extend(values)

    def backtrack() -> bool:
        if csp.is_complete(assignment):
            return True

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                # Propago restricciones en cadena
                removals, success = ac3_propagate(var)
                if success:
                    if backtrack():
                        return True
                restore_domains(removals)
                csp.unassign(var, assignment)

        return False

    if backtrack():
        return assignment
    return None


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking con Forward Checking + heuristicas MRV y LCV.
    - MRV: elijo primero la entrega con menos drones disponibles
      (la mas dificil), para detectar fallos rapido.
    - LCV: pruebo primero el dron que menos restringe a las demas
      entregas, para maximizar las opciones restantes.
    """
    assignment: dict[str, str] = {}

    def select_mrv_variable() -> str:
        """
        Selecciono la entrega sin asignar que tiene menos drones posibles.
        Si hay empate, elijo la que tiene mas vecinos sin asignar (mas restricciones).
        """
        unassigned = csp.get_unassigned_variables(assignment)
        best = None
        best_domain_size = float('inf')
        best_degree = -1

        for var in unassigned:
            domain_size = len(csp.domains[var])
            # Grado: cuantas entregas vecinas aun no tienen dron
            degree = sum(1 for n in csp.get_neighbors(var) if n not in assignment)

            if (domain_size < best_domain_size or
                    (domain_size == best_domain_size and degree > best_degree)):
                best = var
                best_domain_size = domain_size
                best_degree = degree

        return best

    def order_lcv(var: str) -> list[str]:
        """
        Ordeno los drones del que menos restringe al que mas restringe.
        Asi pruebo primero el dron que deja mas opciones a las demas entregas.
        """
        values_with_conflicts = []
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                conflicts = csp.get_num_conflicts(var, value, assignment)
                values_with_conflicts.append((conflicts, value))
        # Ordeno de menor a mayor conflictos
        values_with_conflicts.sort(key=lambda x: x[0])
        return [v for _, v in values_with_conflicts]

    def forward_check(var: str, value: str) -> tuple[dict[str, list[str]], bool]:
        """Igual que en backtracking_fc: elimino opciones invalidas de vecinos."""
        removals: dict[str, list[str]] = {}
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue
            removed: list[str] = []
            for val in csp.domains[neighbor][:]:
                if not csp.is_consistent(neighbor, val, assignment):
                    csp.domains[neighbor].remove(val)
                    removed.append(val)
            if removed:
                removals[neighbor] = removed
            if not csp.domains[neighbor]:
                return removals, False
        return removals, True

    def restore_domains(removals: dict[str, list[str]]) -> None:
        """Restauro los valores eliminados."""
        for neighbor, values in removals.items():
            csp.domains[neighbor].extend(values)

    def backtrack() -> bool:
        if csp.is_complete(assignment):
            return True

        # Elijo la entrega mas restringida (MRV)
        var = select_mrv_variable()
        # Ordeno los drones del menos al mas restrictivo (LCV)
        ordered_values = order_lcv(var)

        for value in ordered_values:
            csp.assign(var, value, assignment)
            removals, success = forward_check(var, value)
            if success:
                if backtrack():
                    return True
            restore_domains(removals)
            csp.unassign(var, assignment)

        return False

    if backtrack():
        return assignment
    return None
