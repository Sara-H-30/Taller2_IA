import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    coverage, redundancy, exposure = problem.score_components(configuration)
    return coverage - redundancy - exposure


def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    # TODO: Add your code here
    raise NotImplementedError("Punto 1: implemente hill_climbing")


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    return initial_temperature * (cooling_rate ** iteration)


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9
    current = tuple(initial_configuration)
    current_score = configuration_score(problem, current)
    evaluations = 1
    iterations = 0
    best_configuration = current
    best_score = current_score
    history: list[Configuration] = [current]
    score_history: list[float] = [current_score]
    for step in range(max_iterations):
        temperature = cooling_schedule(initial_temperature, cooling_rate, step)
        if temperature <= minimum_temperature:
            break
        neighbors = problem.neighbors(current)
        if not neighbors:
            break
        candidate = rng.choice(neighbors)
        candidate_score = configuration_score(problem, candidate)
        evaluations += 1
        iterations += 1
        delta = candidate_score - current_score
        if delta > 0:
            accepted = True
        else:
            accepted = rng.random() < math.exp(delta / temperature)

        if accepted:
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_configuration = current
                best_score = current_score

        history.append(current)
        score_history.append(current_score)

    return OptimizationResult(
        best_configuration=best_configuration,
        best_score=best_score,
        evaluations=evaluations,
        iterations=iterations,
        history=history,
        score_history=score_history,
    )
    

def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    cut = rng.randint(1, len(parent1) - 1)
    child1 = tuple(parent1[:cut]) + tuple(parent2[cut:])
    child2 = tuple(parent2[:cut]) + tuple(parent1[cut:])
    return child1, child2


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    if rng.random() >= mutation_probability:
        return tuple(individual)

    active = [index for index, bit in enumerate(individual) if bit]
    inactive = [index for index, bit in enumerate(individual) if not bit]
    if not active or not inactive:
        return tuple(individual)

    remove = rng.choice(active)
    add = rng.choice(inactive)
    mutated = list(individual)
    mutated[remove] = 0
    mutated[add] = 1
    return tuple(mutated)

def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    population = problem.initial_population(population_size, rng)
    scores = [configuration_score(problem, individual) for individual in population]
    evaluations = len(population)

    best_index = max(range(len(population)), key=lambda index: scores[index])
    best_configuration = population[best_index]
    best_score = scores[best_index]

    history: list[Configuration] = [best_configuration]
    score_history: list[float] = [best_score]

    offspring_needed = population_size - elite_size

    for _ in range(generations):
        # Elitismo: los mejores individuos pasan intactos a la nueva población.
        ranking = sorted(range(len(population)), key=lambda index: scores[index], reverse=True)
        elite_indices = ranking[:elite_size]
        new_population = [population[index] for index in elite_indices]
        new_scores = [scores[index] for index in elite_indices]

        offspring: list[Configuration] = []
        while len(offspring) < offspring_needed:
            parent1 = problem.tournament_select(population, scores, rng)
            parent2 = problem.tournament_select(population, scores, rng)
            child1, child2 = one_point_crossover(parent1, parent2, rng)
            for child in (child1, child2):
                if len(offspring) >= offspring_needed:
                    break
                repaired = problem.repair_configuration(child, rng)
                mutated = swap_mutation(repaired, mutation_probability, rng)
                offspring.append(mutated)

        for child in offspring:
            new_population.append(child)
            new_scores.append(configuration_score(problem, child))
            evaluations += 1

        population = new_population
        scores = new_scores

        generation_best = max(range(len(population)), key=lambda index: scores[index])
        if scores[generation_best] > best_score:
            best_configuration = population[generation_best]
            best_score = scores[generation_best]

        history.append(best_configuration)
        score_history.append(best_score)

    return OptimizationResult(
        best_configuration=best_configuration,
        best_score=best_score,
        evaluations=evaluations,
        iterations=generations,
        history=history,
        score_history=score_history,
    )
