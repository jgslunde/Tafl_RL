#!/usr/bin/env python3
"""
Plot how well the policy of a run finds forced wins, as a function of the training iteration.

The positions come from tactics/<rule set>.jsonl (see analysis/tactics_database.py): 20 positions
per win condition and depth, where depth 1 is a win at once and depth 2 is a forced win (move, any
reply, winning move). A run's rule set is read from its run_config.json. A rule set without such a
file is skipped.

Every chosen checkpoint is run on all positions with probe_tactics.probe_network, without search.
The means per condition and depth go to tactics_history.jsonl in the run directory, so that a later
call computes only new checkpoints. That file stores a hash of the positions: if the positions
change, the old results no longer compare, and everything is computed again.

Plots:
  plot_tactics.png             In the run directory: the policy share of the winning moves per
                               iteration, one panel per depth. Dashed lines show what a policy
                               without knowledge gives the same moves. The axis is logarithmic,
                               since a rise from 0.02 to 0.05 matters as much as one from 0.4 to 1.
  plot_tactics_comparison.png  With several runs, next to the run directories: one panel per
                               condition and depth, one line per run.

Usage: python plot_tactics.py <run directory> [more run directories ...] [--every N]

Example: python plot_tactics.py checkpoints/copenhagen/baseline
Example: python plot_tactics.py checkpoints/copenhagen/baseline \
             checkpoints/copenhagen/line_10x64_gpool
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot import ENDING_COLORS
from probe_tactics import probe_network
from tafl import PIECE_CODES, RULESETS, Tafl

HERE = os.path.dirname(os.path.abspath(__file__))
# Groups that say nothing about the networks: attackers_blocked never happens in games, and the
# history of the depth 2 repetitions is invented, while the networks cannot see the history
SKIPPED = {(1, 'attackers_blocked'), (2, 'attackers_blocked'), (2, 'repetition')}
COLORS = {**ENDING_COLORS, 'encircled_pocket': '#e8873a'}
NAMES = {'king_escaped': 'King escape', 'exit_fort': 'Exit fort', 'king_captured': 'King capture',
         'encircled': 'Proper ring', 'encircled_pocket': 'Lone-king pocket',
         'defenders_blocked': 'Defenders blocked', 'repetition': 'Repetition',
         'attackers_blocked': 'Attackers blocked'}


def load_positions(path: str, rules) -> tuple:
    """The positions of a database file, as (their dicts, a game for each, its set of winning
    moves for each, as probe_network wants them)."""
    positions, probes = [], []
    for line in open(path):
        position = json.loads(line)
        game = Tafl(rules)
        board = [[PIECE_CODES[c] for c in row] for row in position['board']]
        game.board = np.array(board, dtype=game.board.dtype)
        game.current_player = position['player']
        game.position_history = [game._position_key()]
        if position['history']:  # each entry is the board as text, then the side to move
            game.position_history = []
            for text in position['history']:
                earlier = np.array([PIECE_CODES[c] for c in text[:-1]], dtype=game.board.dtype)
                earlier = earlier.reshape(game.board.shape)
                game.position_history.append((earlier.tobytes(), int(text[-1])))
        positions.append(position)
        probes.append((game, {tuple(move) for move in position['winning_moves']}))
    return positions, probes


def update_history(run_directory: str, every: int) -> list:
    """Measure the checkpoints of a run that have no results yet, and return all results as a list
    of {'iteration', 'groups': {'<depth> <condition>': means}}, by iteration. Returns None if the
    rule set has no database."""
    with open(os.path.join(run_directory, 'run_config.json')) as f:
        ruleset = json.load(f)['game']['ruleset']
    database = os.path.join(HERE, 'tactics', f'{ruleset}.jsonl')
    if not os.path.exists(database):
        print(f'{run_directory}: no database for {ruleset}, skipped')
        return None
    with open(database, 'rb') as f:
        database_hash = hashlib.sha256(f.read()).hexdigest()

    history_path = os.path.join(run_directory, 'tactics_history.jsonl')
    rows = []
    if os.path.exists(history_path):
        rows = [json.loads(line) for line in open(history_path)]
        rows = [row for row in rows if row['database'] == database_hash]

    # All checkpoints at multiples of `every`, and always the newest one
    paths = {int(re.search(r'checkpoint_iter_(\d+)\.pth$', path)[1]): path
             for path in glob.glob(os.path.join(run_directory, 'checkpoint_iter_*.pth'))}
    chosen = [i for i in sorted(paths) if i % every == 0 or i == max(paths)]
    done = {row['iteration'] for row in rows}
    missing = [iteration for iteration in chosen if iteration not in done]
    print(f'{run_directory}: {len(done)} checkpoints done, {len(missing)} to measure')

    if missing:
        positions, probes = load_positions(database, RULESETS[ruleset])
        conditions = [p['condition'] for p in positions]
        endings = ['encircled' if name == 'encircled_pocket' else name for name in conditions]
        labels = np.array([f"{p['depth']} {p['condition']}" for p in positions])
    # Rewritten in full, as results of an older database must go
    with open(history_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
        for iteration in missing:
            results = probe_network(paths[iteration], probes, endings, 0)
            groups = {label: {key: float(results[key][labels == label].mean())
                              for key in ('uniform', 'p_win', 'top1', 'value', 'ending')}
                      for label in np.unique(labels)}
            rows.append({'iteration': iteration, 'database': database_hash, 'groups': groups})
            f.write(json.dumps(rows[-1]) + '\n')
            f.flush()
            print(f'  iteration {iteration}', flush=True)
    return sorted(rows, key=lambda row: row['iteration'])


def group_keys(rows: list) -> list:
    """The (depth, condition) of all groups to plot, in the order of the database's conditions."""
    keys = [(int(label.split()[0]), label.split()[1]) for label in rows[0]['groups']]
    keys = [key for key in keys if key not in SKIPPED]
    return sorted(keys, key=lambda key: (key[0], list(NAMES).index(key[1])))


def plot_run(run_directory: str, rows: list):
    """The policy share of the winning moves per iteration, one panel per depth."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    iterations = [row['iteration'] for row in rows]
    for depth, ax, title in [(1, axes[0], 'Win in 1'), (2, axes[1], 'Forced Win in 2')]:
        for key in [key for key in group_keys(rows) if key[0] == depth]:
            label, color = f'{depth} {key[1]}', COLORS[key[1]]
            ax.plot(iterations, [row['groups'][label]['p_win'] for row in rows], 'o-', color=color,
                    linewidth=2, markersize=4, label=NAMES[key[1]])
            ax.axhline(y=rows[-1]['groups'][label]['uniform'], color=color, linestyle='--',
                       linewidth=1, alpha=0.6)
        ax.set_yscale('log')
        ax.set_ylim(0.005, 1.1)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('Policy Share of Winning Moves', fontsize=12)
    # One legend for both panels, outside them. Depth 1 has all conditions that are plotted.
    fig.legend(*axes[0].get_legend_handles_labels(), loc='center right', fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    plt.savefig(os.path.join(run_directory, 'plot_tactics.png'), dpi=150)
    plt.close()


def plot_comparison(histories: dict, path: str):
    """One panel per condition and depth, one line per run (histories: {run directory: rows})."""
    keys = group_keys(next(iter(histories.values())))
    columns = sorted({condition for _, condition in keys}, key=list(NAMES).index)
    fig, axes = plt.subplots(2, len(columns), figsize=(3.6 * len(columns), 7.5), sharex=True,
                             sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for depth, condition in keys:
        ax = axes[depth - 1, columns.index(condition)]
        ax.set_visible(True)
        label = f'{depth} {condition}'
        for run_directory, rows in histories.items():
            iterations = [row['iteration'] for row in rows]
            ax.plot(iterations, [row['groups'][label]['p_win'] for row in rows], 'o-',
                    linewidth=1.5, markersize=3, label=os.path.basename(run_directory.rstrip('/')))
        ax.axhline(y=rows[-1]['groups'][label]['uniform'], color='gray', linestyle='--',
                   linewidth=1)
        ax.set_yscale('log')
        ax.set_ylim(0.005, 1.1)
        ax.set_title(f"{NAMES[condition]}, {'win in 1' if depth == 1 else 'forced in 2'}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, which='both', alpha=0.3)
    # One legend for all panels, from a panel that has every run
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=len(histories),
               fontsize=10)
    fig.supylabel('Policy Share of Winning Moves', fontsize=12)
    plt.tight_layout(rect=[0.015, 0.04, 1, 1])
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_directories', nargs='+')
    parser.add_argument('--every', type=int, default=10,
                        help='measure every N-th iteration (the newest one always)')
    args = parser.parse_args()

    histories = {}
    for run_directory in args.run_directories:
        rows = update_history(run_directory, args.every)
        if rows:
            histories[run_directory] = rows
            plot_run(run_directory, rows)
            print(f"  ✓ Saved {os.path.join(run_directory, 'plot_tactics.png')}")
    if len(histories) > 1:
        parent = os.path.dirname(args.run_directories[0].rstrip('/'))
        path = os.path.join(parent, 'plot_tactics_comparison.png')
        plot_comparison(histories, path)
        print(f'  ✓ Saved {path}')


if __name__ == '__main__':
    main()
