#!/usr/bin/env python3
"""
Plot statistics of the played games of a run from its games.jsonl.

train.py appends one line to games.jsonl for every finished game (see game_record in train.py).
Games of type vs_random are not plotted. The rule set is read from the run_config.json next to the
input file. The king never counts as a defender.

Plots, saved next to the input file:
  plot_pieces_left.png             Mean number of attackers and defenders left on the final board
                                   when each side won, per iteration, with the numbers at the start
  plot_pieces_vs_moves.png         Pieces left on the final board vs game length, for each game of
                                   the last 40 iterations
  plot_attackers_vs_defenders.png  Attackers left vs defenders left on the final board, for each
                                   game with a winner of the last 40 iterations, in one panel per
                                   winner and kind of game
  plot_final_boards_selfplay.png   Final boards of 28 random self-play games and of 28 random
  plot_final_boards_vs_best.png    evaluation games of the last 40 iterations, with the color of
                                   the winner around each board

Self-play and the new-vs-best evaluation have separate panels or files. Random numbers have a fixed
seed, so the same file always gives the same plots.

A resumed run writes the iterations it repeats to games.jsonl again, so those iterations hold the
games of both tries.

Usage: python plot_games.py <games.jsonl>

Example: python plot_games.py checkpoints/brandubh_2/baseline/games.jsonl
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from tafl import RULESETS, Rules, Tafl


LAST_ITERATIONS = 40  # the scatter plots and the final boards use the games of this many iterations
ATTACKER_COLOR, DEFENDER_COLOR, KING_COLOR, DRAW_COLOR = '#a32020', '#1f4e8c', '#e8b90e', '#8c8c8c'
PANELS = [('selfplay', 'Self-Play Games'), ('vs_best', 'Evaluation Games (New vs Best Network)')]
SIDES = [('A', 'Attackers', ATTACKER_COLOR), ('D', 'Defenders', DEFENDER_COLOR)]


def pieces_left(games: list[dict], piece: str) -> np.ndarray:
    """The number of pieces of type `piece` ('A' or 'D') on the final board of each game."""
    return np.array([sum(row.count(piece) for row in game['board']) for game in games])


def plot_pieces_left(games: list[dict], rules: Rules, output_dir: str):
    """Mean number of pieces of each side left on the final board, per iteration, for the games
    won by each side (draws have no winner and are left out). The color shows the side of the
    counted pieces, the line style the winner. Dashed lines show the numbers at the start."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for ax, (game_type, title) in zip(axes, PANELS):
        for winner, style in [('attackers', 'o-'), ('defenders', 's:')]:
            wins = [game for game in games if (game['type'], game['winner']) == (game_type, winner)]
            game_iterations = np.array([game['iteration'] for game in wins])
            iterations = np.unique(game_iterations)  # only iterations where this side won a game
            for piece, name, color in SIDES:
                left = pieces_left(wins, piece)
                means = [left[game_iterations == iteration].mean() for iteration in iterations]
                ax.plot(iterations, means, style, color=color, linewidth=2, markersize=4,
                        label=f'{name} left, {winner} won')
        for piece, name, color in SIDES:
            start_count = ''.join(rules.setup).count(piece)
            ax.axhline(y=start_count, color=color, linestyle='--', linewidth=1,
                       label=f'{name} at start')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Mean Pieces Left on Final Board', fontsize=12)

    # One legend for both panels, outside them so that it covers no lines
    fig.legend(*axes[0].get_legend_handles_labels(), loc='center right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.81, 1])
    plt.savefig(os.path.join(output_dir, 'plot_pieces_left.png'), dpi=150)
    plt.close()


def plot_pieces_vs_moves(games: list[dict], output_dir: str):
    """Pieces of each side left on the final board vs the length of the game, one point per side
    and game (draws included)."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

    for ax, (game_type, title) in zip(axes, PANELS):
        panel_games = [game for game in games if game['type'] == game_type]
        plies = np.array([game['plies'] for game in panel_games])
        for piece, _, color in SIDES:
            # Piece counts are whole numbers: a small random shift shows how many points share one
            left = pieces_left(panel_games, piece) + rng.uniform(-0.3, 0.3, len(panel_games))
            ax.scatter(plies, left, s=6, color=color, alpha=0.2, linewidths=0)
        ax.set_xlabel('Game Length (plies)', fontsize=12)
        ax.set_title(f'{title}, n={len(panel_games)}', fontsize=13, fontweight='bold')
        ax.set_ylim(bottom=-0.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Pieces Left on Final Board', fontsize=12)

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, label=f'{name} left')
               for _, name, color in SIDES]
    fig.legend(handles=handles, loc='center right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(os.path.join(output_dir, 'plot_pieces_vs_moves.png'), dpi=150)
    plt.close()


def plot_attackers_vs_defenders(games: list[dict], output_dir: str):
    """Attackers left vs defenders left on the final board, one point per game. One row of panels
    for self-play and one for the evaluation, one column for each winner. Draws are left out."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    for row, (game_type, title) in zip(axes, PANELS):
        for ax, winner, color in [(row[0], 'attackers', ATTACKER_COLOR),
                                  (row[1], 'defenders', DEFENDER_COLOR)]:
            wins = [game for game in games if (game['type'], game['winner']) == (game_type, winner)]
            # Piece counts are whole numbers: a small random shift shows how many points share one
            defenders = pieces_left(wins, 'D') + rng.uniform(-0.3, 0.3, len(wins))
            attackers = pieces_left(wins, 'A') + rng.uniform(-0.3, 0.3, len(wins))
            ax.scatter(defenders, attackers, s=8, color=color, alpha=0.3, linewidths=0)
            ax.set_title(f'{title}\n{winner.capitalize()} won, n={len(wins)}', fontsize=12,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Defenders Left on Final Board', fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel('Attackers Left on Final Board', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_attackers_vs_defenders.png'), dpi=150)
    plt.close()


def plot_final_boards(games: list[dict], rules: Rules, game_type: str, title: str, output_dir: str):
    """Final boards of 28 random games of type `game_type`, on 4 rows of 7 boards. The border of
    each board has the color of the winner, gray for a draw."""
    rng = np.random.default_rng(0)
    candidates = [game for game in games if game['type'] == game_type]
    chosen = rng.choice(len(candidates), size=min(28, len(candidates)), replace=False)
    border_colors = {'attackers': ATTACKER_COLOR, 'defenders': DEFENDER_COLOR, None: DRAW_COLOR}

    n = rules.board_size
    game_state = Tafl(rules)
    special = np.zeros((n, n))  # 1 on the corners and the throne, where they exist
    for square in game_state.corners + ([game_state.throne] if rules.throne else []):
        special[square] = 1
    marker_size = (130 / n)**2  # marker area in points^2: the piece diameter shrinks with n

    fig, axes = plt.subplots(4, 7, figsize=(22, 14.5))
    for ax in axes.flat:
        ax.axis('off')  # also for panels left empty if there are fewer than 28 games
    for ax, index in zip(axes.flat, chosen):
        game = candidates[index]
        board = np.array([list(row) for row in game['board']])
        ax.pcolormesh(special, cmap='Greys', vmin=0, vmax=2, edgecolors='#b0b0b0', linewidth=0.5)
        for piece, color in [('A', ATTACKER_COLOR), ('D', DEFENDER_COLOR), ('K', KING_COLOR)]:
            rows, columns = np.where(board == piece)
            ax.scatter(columns + 0.5, rows + 0.5, s=marker_size, color=color, edgecolors='black',
                       linewidths=0.5)
        ax.add_patch(plt.Rectangle((0, 0), n, n, fill=False, linewidth=5, clip_on=False,
                                   edgecolor=border_colors[game['winner']]))
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)  # row 0 at the top, as in the setups of tafl.py
        ax.set_aspect('equal')
        result = 'Draw' if game['winner'] is None else f"{game['winner'].capitalize()} won"
        ax.set_title(f"{result}: {game['ending']}\n"
                     f"iteration {game['iteration']}, {game['plies']} plies", fontsize=10)

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, markeredgecolor='black',
                          label=name) for color, name in [(ATTACKER_COLOR, 'Attacker'),
                                                          (DEFENDER_COLOR, 'Defender'),
                                                          (KING_COLOR, 'King')]]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11,
               title='Border: red for attacker wins, blue for defender wins, gray for draws')
    fig.suptitle(f'{title}: Final Boards of Random Games', fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.97], w_pad=3, h_pad=3)
    plt.savefig(os.path.join(output_dir, f'plot_final_boards_{game_type}.png'), dpi=150)
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_games.py <games.jsonl>")
        sys.exit(1)
    filepath = sys.argv[1]
    output_dir = os.path.dirname(filepath) or '.'

    print(f"Reading {filepath}")
    with open(filepath, 'r') as f:
        games = [json.loads(line) for line in f]
    with open(os.path.join(output_dir, 'run_config.json'), 'r') as f:
        rules = RULESETS[json.load(f)['game']['ruleset']]

    last_iteration = max(game['iteration'] for game in games)
    recent_games = [game for game in games if game['iteration'] > last_iteration - LAST_ITERATIONS]

    print(f"\nCreating plots...")
    print(f"  - {len(games)} games, {len(recent_games)} in the last {LAST_ITERATIONS} iterations")

    plot_pieces_left(games, rules, output_dir)
    print("  ✓ Saved plot_pieces_left.png")
    plot_pieces_vs_moves(recent_games, output_dir)
    print("  ✓ Saved plot_pieces_vs_moves.png")
    plot_attackers_vs_defenders(recent_games, output_dir)
    print("  ✓ Saved plot_attackers_vs_defenders.png")
    for game_type, title in PANELS:
        plot_final_boards(recent_games, rules, game_type, title, output_dir)
        print(f"  ✓ Saved plot_final_boards_{game_type}.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
