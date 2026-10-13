"""Cost-based search algorithms for the Pokémon environment."""

from __future__ import annotations

import heapq
import math
import random
import matplotlib.pyplot as plt
from itertools import count

from pokemon_game import (
    GameMap,
    State,
    movement_cost,
    successors,
)


def reconstruct_path(
    parents: dict[State, State | None],
    goal: State,
) -> list[State]:
    """Reconstruct a path from a parent dictionary."""
    path: list[State] = []
    current: State | None = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


def path_cost(
    game_map: GameMap,
    path: list[State],
) -> int:
    """Return the total cost of a path."""
    return sum(
        movement_cost(game_map, state)
        for state in path[1:]
    )


def manhattan_distance(
    state: State,
    goal: State,
) -> int:
    """Return the Manhattan distance between two states."""
    return (
        abs(state[0] - goal[0])
        + abs(state[1] - goal[1])
    )


def euclidean_distance(
    state: State,
    goal: State,
) -> float:
    """Return the Euclidean distance between two states."""
    return math.hypot(
        state[0] - goal[0],
        state[1] - goal[1],
    )


def minkowski_distance(
    state: State,
    goal: State,
    p: float = 1,
) -> float:
    """Return the Minkowski distance of order p between two states.

    p = 1 gives the Manhattan distance, p = 2 the Euclidean distance.
    """
    return (
        abs(state[0] - goal[0]) ** p
        + abs(state[1] - goal[1]) ** p
    ) ** (1 / p)


def generate_random_path(
    game_map: GameMap,
    start: State,
    goal: State,
    seed: int | None = None,
) -> list[State]:
    """Generate a random valid path from start to goal.

    This function is provided only to demonstrate the visualization.
    It is not guaranteed to return a minimum-cost or shortest path.

    The function performs a randomized depth-first exploration: the valid
    successor states are shuffled before being added to the frontier.

    Parameters
    ----------
    game_map : GameMap
        Weighted Pokémon map.
    start : State
        Initial trainer position.
    goal : State
        Pokémon Center position.
    seed : int or None, default=None
        Random seed used to make the result reproducible.

    Returns
    -------
    list[State]
        A valid path beginning at ``start`` and ending at ``goal``.

    Raises
    ------
    RuntimeError
        If no path exists between the start and the goal.
    """
    rng = random.Random(seed)

    frontier = [start]

    parents: dict[State, State | None] = {
        start: None,
    }

    while frontier:
        current = frontier.pop()

        if current == goal:
            return reconstruct_path(parents, goal)

        next_states = successors(game_map, current)
        rng.shuffle(next_states)

        for next_state in next_states:
            if next_state in parents:
                continue

            parents[next_state] = current
            frontier.append(next_state)

    raise RuntimeError(
        "No path exists between the start and the goal."
    )


def dijkstra(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using Dijkstra's algorithm."""
    ordre_insertion = count()
    file_priorite = [(0, next(ordre_insertion), start)]
    cout = {start: 0}
    parents = {start: None}
    deja_traites = set()

    while file_priorite:
        _, _, etat = heapq.heappop(file_priorite)

        if etat in deja_traites:
            continue
        deja_traites.add(etat)

        if etat == goal:
            return reconstruct_path(parents, goal), len(deja_traites)

        for suivant in successors(game_map, etat):
            nouveau_cout = cout[etat] + movement_cost(game_map, suivant)
            if suivant not in cout or nouveau_cout < cout[suivant]:
                cout[suivant] = nouveau_cout
                parents[suivant] = etat
                heapq.heappush(
                    file_priorite,
                    (nouveau_cout, next(ordre_insertion), suivant),
                )

    return [], len(deja_traites)


def a_star(
    game_map: GameMap,
    start: State,
    goal: State,
    p: float = 1,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using A* search (Minkowski heuristic)."""
    ordre_insertion = count()
    valeur_f_depart = minkowski_distance(start, goal, p)
    file_priorite = [(valeur_f_depart, next(ordre_insertion), start)]
    cout = {start: 0}
    parents = {start: None}
    deja_traites = set()

    while file_priorite:
        _, _, etat = heapq.heappop(file_priorite)

        if etat in deja_traites:
            continue
        deja_traites.add(etat)

        if etat == goal:
            return reconstruct_path(parents, goal), len(deja_traites)

        for suivant in successors(game_map, etat):
            nouveau_cout = cout[etat] + movement_cost(game_map, suivant)
            if suivant not in cout or nouveau_cout < cout[suivant]:
                cout[suivant] = nouveau_cout
                parents[suivant] = etat
                valeur_f = nouveau_cout + minkowski_distance(suivant, goal, p)
                heapq.heappush(
                    file_priorite,
                    (valeur_f, next(ordre_insertion), suivant),
                )

    return [], len(deja_traites)


def plot_states_vs_p(
    game_map: GameMap,
    start: State,
    goal: State,
    p_max: int = 10,
) -> None:
    """Plot the number of states considered by A* for p = 1..p_max."""

    valeurs_p = list(range(1, p_max + 1))
    nb_etats = []

    for p in valeurs_p:
        _, nb = a_star(game_map, start, goal, p=p)
        nb_etats.append(nb)
        print(f"p = {p:2d} -> {nb} états considérés")

    _, nb_dijkstra = dijkstra(game_map, start, goal)

    plt.figure(figsize=(8, 5))
    plt.plot(valeurs_p, nb_etats, marker="o", label="A* (Minkowski)")
    plt.axhline(
        nb_dijkstra,
        color="red",
        linestyle="--",
        label=f"Dijkstra ({nb_dijkstra})",
    )
    plt.xlabel("p")
    plt.ylabel("Nombre d'états considérés")
    plt.title("A* : états considérés selon l'ordre p de Minkowski")
    plt.xticks(valeurs_p)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
