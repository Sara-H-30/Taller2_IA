import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


# Pesos de la función de evaluación
PESO_TERMINAL_PENDIENTE = 60.0   # castigo por cada terminal que falta activar
PESO_DISTANCIA_OBJETIVO = 4.0    # castigo por cada paso hasta la terminal objetivo
PESO_DISTANCIA_INTRUSO = 2.0     # premio por cada paso de separación del intruso
TOPE_DISTANCIA_INTRUSO = 6       # más allá de esta distancia el intruso ya no preocupa
CASTIGO_CAPTURA_INMEDIATA = 100.0  # intruso a 1 paso o menos del defensor
PESO_MOVILIDAD = 1.0             # premio por cada acción legal del defensor


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    layout = state.layout
    defensor = state.defender_position
    intruso = state.intruder_position

    # Número grande que se usa en lugar de "infinito" cuando no hay camino
    distancia_grande = layout.width * layout.height

    # 1) Terminal objetivo del defensor.
    # Una terminal es "segura" si el defensor llega estrictamente antes que el
    # intruso. Se prefiere la segura más cercana. Si no hay ninguna, se usa la
    # más cercana de todas.
    distancia_segura = distancia_grande
    distancia_cualquiera = distancia_grande
    for terminal in state.pending_terminals:
        distancia_defensor = layout.distance(defensor, terminal)
        distancia_del_intruso = layout.distance(intruso, terminal)

        if distancia_defensor < distancia_cualquiera:
            distancia_cualquiera = distancia_defensor

        if distancia_defensor < distancia_del_intruso:
            if distancia_defensor < distancia_segura:
                distancia_segura = distancia_defensor

    if distancia_segura < distancia_grande:
        distancia_objetivo = distancia_segura
    else:
        distancia_objetivo = distancia_cualquiera

    # 2) Distancia real del intruso al defensor (con tope)
    distancia_intruso = layout.distance(intruso, defensor)
    if distancia_intruso > TOPE_DISTANCIA_INTRUSO:
        distancia_intruso = TOPE_DISTANCIA_INTRUSO

    # 3) Movilidad: cuántas acciones legales tiene el defensor
    movilidad = len(state.get_legal_actions(0))

    # Suma ponderada de las características
    valor = state.get_score()
    valor = valor - PESO_TERMINAL_PENDIENTE * len(state.pending_terminals)
    valor = valor - PESO_DISTANCIA_OBJETIVO * distancia_objetivo
    valor = valor + PESO_DISTANCIA_INTRUSO * distancia_intruso
    if distancia_intruso <= 1:
        valor = valor - CASTIGO_CAPTURA_INMEDIATA
    valor = valor + PESO_MOVILIDAD * movilidad

    # Garantiza que ningún estado no terminal llegue a +1000 o -1000
    if valor > 999.0:
        valor = 999.0
    if valor < -999.0:
        valor = -999.0
    return float(valor)