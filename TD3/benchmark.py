import matplotlib.pyplot as plt

from algorithms import alphabeta, mcts, minimax
from game import create_board

FUNCTIONS = {"minimax": minimax, "alphabeta": alphabeta, "mcts": mcts}


def benchmark(board, algorithm, **kwargs):
    result = FUNCTIONS[algorithm](board, **kwargs)
    return {
        "name": algorithm,
        "move": result["move"],
        "score": result["score"],
        "nodes": result["nodes_explored"],
        "time": result["time"],
    }


if __name__ == "__main__":
    board = create_board()
    depths = [1, 2, 3, 4, 5]
    simulations = [100, 500, 1000, 2000]

    nodes_minimax = [benchmark(board, "minimax", depth=d)["nodes"] for d in depths]
    nodes_alphabeta = [benchmark(board, "alphabeta", depth=d)["nodes"] for d in depths]
    times_mcts = [benchmark(board, "mcts", num_simulations=n, seed=0)["time"] for n in simulations]

    plt.figure()
    plt.plot(depths, nodes_minimax, marker="o", label="Minimax")
    plt.plot(depths, nodes_alphabeta, marker="s", label="Alpha-Beta")
    plt.yscale("log")
    plt.xlabel("Profondeur")
    plt.ylabel("Noeuds explorés")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("nodes_vs_depth.png", dpi=150)

    plt.figure()
    plt.plot(simulations, times_mcts, marker="o", label="MCTS")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Temps (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mcts_time_vs_simulations.png", dpi=150)

    plt.show()
