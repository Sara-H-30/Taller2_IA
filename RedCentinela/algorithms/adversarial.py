from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        # Cada llamada a get_action empieza con el contador en cero
        self.nodes_evaluated = 0

        # La raíz también es un estado procesado, así que se cuenta
        self.nodes_evaluated = self.nodes_evaluated + 1

        # Si el juego ya terminó en la raíz, no hay acción que escoger
        if state.is_win() or state.is_lose():
            return None

        acciones = state.get_legal_actions(0)
        if len(acciones) == 0:
            return None

        # En la raíz juega el defensor (agente 0, MAX)
        siguiente_agente = 1 % state.get_num_agents()

        mejor_accion = None
        mejor_valor = float("-inf")

        for accion in acciones:
            sucesor = state.generate_successor(0, accion)
            
            valor = self.minimax_value(sucesor, siguiente_agente, self.depth - 1)

            
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion

    def minimax_value(self, state, agente, profundidad_restante):
        """
        Retorna el valor Minimax de un estado.

        agente: índice del agente que juega en este estado (0 = MAX, 1 = MIN).
        profundidad_restante: cuántos plies quedan por explorar.
        """
        
        self.nodes_evaluated = self.nodes_evaluated + 1

        # Caso base 1: estado terminal (el defensor ganó o fue interceptado)
        if state.is_win() or state.is_lose():
            return evaluation_function(state)

        # Caso base 2: se acabó la profundidad permitida (corte)
        if profundidad_restante == 0:
            return evaluation_function(state)

        acciones = state.get_legal_actions(agente)

        # Caso base 3: el agente no tiene movimientos posibles
        if len(acciones) == 0:
            return evaluation_function(state)

        siguiente_agente = (agente + 1) % state.get_num_agents()

        if agente == 0:
            # Nodo MAX (defensor): se queda con el valor más alto
            mejor = float("-inf")
            for accion in acciones:
                sucesor = state.generate_successor(agente, accion)
                valor = self.minimax_value(sucesor, siguiente_agente, profundidad_restante - 1)
                if valor > mejor:
                    mejor = valor
            return mejor
        else:
            # Nodo MIN (intruso): se queda con el valor más bajo para MAX
            peor = float("inf")
            for accion in acciones:
                sucesor = state.generate_successor(agente, accion)
                valor = self.minimax_value(sucesor, siguiente_agente, profundidad_restante - 1)
                if valor < peor:
                    peor = valor
            return peor


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        # TODO: Add your code here
        raise NotImplementedError("Punto 5: implemente AlphaBetaAgent.get_action")