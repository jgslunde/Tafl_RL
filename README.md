# Tafl RL

AlphaZero-style self-play training for Tafl games: Brandubh on 7x7, Tablut variants on 9x9 and
Copenhagen Hnefatafl and its relatives on 11x11. Everything runs on CPU nodes. Self-play happens in
many single-threaded worker processes, and the network is trained with PyTorch in the main
process. Trained networks can be exported to ONNX and played in the browser, in the separate
`HnefataflOnlineGame` repository.

The networks learn from self-play and the rules only. Human games are never used for training;
they are used afterwards, to see how the networks compare with human play.

Other documents in the repository:

- `RULESETS.md` lists the named rule sets, where their rules come from and how balanced they are.
- `tafl_variants_overview_updated.md` describes the most-played variants in detail. `tafl.py`
  implements its rules.
- `NETWORK_ARCHITECTURES.md` explains the network designs and has a table of every training run
  and what it changed.
- `FUTURE_IMPROVEMENTS.md` collects ideas that have not been tried yet. It was written in October
  2026, and a few of its items have been done since.
- `analysis/README.md` covers the opening books and other analyses of trained networks.


## Getting started

Use Python 3.13. Python 3.14 no longer starts pool workers with `fork` by default, which the
self-play stop signal depends on, and some of the dependencies have no 3.14 wheels yet.

    python3.13 -m venv .venv
    .venv/bin/pip install -r requirements.txt

The requirements install the CPU build of PyTorch.

A training run is described by one JSON file, `configs/<rule set>/<run>.json`, and writes
everything to `checkpoints/<rule set>/<run>/`. The usual way to start one is `run.sh`, inside
`screen` so that it survives a logout:

    screen -S cph ./run.sh copenhagen baseline

`run.sh` starts 64 self-play workers and lets training use 64 threads; change the numbers at the
top of the script to fit the node. Any further arguments go to `run_training.py`:

    ./run.sh copenhagen baseline --set self_play.num_iterations=50
    ./run.sh copenhagen baseline --resume checkpoints/copenhagen/baseline/checkpoint_iter_120.pth
    .venv/bin/python run_training.py configs/copenhagen/baseline.json --show

`--show` prints the resolved config and exits. `--resume` continues a run with its network,
optimizer state and training history. The replay buffer is not saved, so a resumed run starts
with an empty buffer and does not train until it holds `min_buffer_size` samples again.
`--load-weights` starts a fresh run from the weights of another network.

To branch a run, write a new config and resume it from a checkpoint of the old run. The iteration
count carries on from the checkpoint, and the new run writes to its own directory.

The tests are plain scripts that exit with code 1 if anything fails:

    .venv/bin/python test_rules.py           # the rule sets, one hand-made position per rule
    .venv/bin/python test_tafl_network.py    # networks, input planes, configs, a short training run
    .venv/bin/python test_training_fixes.py  # search, self-play, replay buffer, training step

Together they take a few minutes.


## Where things are

| File | Contents |
|---|---|
| `tafl.py` | The game engine and the named rule sets (`RULESETS`), the list of game endings (`ENDINGS`) and the policy index of a move (`move_index`) |
| `features.py` | The network's input planes (`FeatureSpec`, `build_features`) and the position history (`StateHistory`) |
| `network_config.py` | `NetworkConfig`, a description of an architecture that does not need torch |
| `tafl_network.py` | `TaflNet`, built from a `NetworkConfig`, and loading of checkpoints |
| `mcts.py` | The tree search |
| `agent.py` | Players built on the search, and `play_game` |
| `train.py` | The training loop: self-play workers, replay buffer, training step, gating, checkpoints |
| `run_config.py` | `TrainingConfig`, with every setting and its default, and the JSON format |
| `run_training.py` | The command line for starting and resuming runs |
| `convert_to_onnx.py` | Export to ONNX, and `OnnxNet`, which the self-play workers run |
| `plot.py`, `plot.sh` | Plots of a run's training history (`plot.sh` does every run) |
| `plot_games.py` | Plots from a run's game records |
| `plot_tactics.py`, `probe_tactics.py` | How well the policy finds forced wins, over the run |
| `evaluate_elo.py`, `plot_elo.py` | Elo ratings from matches between the checkpoints of a run |
| `play.py` | Two checkpoints play each other in the terminal |
| `gui_play.py` | A pygame board: play against a network or watch it play |
| `benchmark_network.py` | Time one network call for the networks of some configs |
| `analysis/` | Opening books, comparisons with human games, the tactics database |
| `human_games/` | Online games from aagenielsen.dk, one file per variant (analysis only) |
| `tactics/` | Positions with forced wins, per rule set, for `plot_tactics.py` |


## One training iteration

An iteration has four steps: self-play, training, an optional match against a random player,
and the gating match against the best network so far. The log marks them `[1/4]` to `[4/4]`.

### Self-play

Before self-play, the main process exports the current network to a temporary ONNX file. Each
worker loads that file into ONNX Runtime on a single thread. For the one-position calls that the
search makes, this is about three times faster than compiled PyTorch, and nothing needs to be
compiled. The pool starts up to twice `num_games_per_iteration` games and stops the rest once
enough have finished. This keeps every worker busy until the end, at the cost of losing a few
long games that were still running.

A game goes like this:

1. In half of the games the first 2, 4, 6 or 8 plies are uniformly random moves
   (`use_random_opening_moves`). They are not stored. They put the network in positions its own
   play would never choose.
2. Before every move the worker checks whether some move wins at once. If one does, it is
   played without a search and stored with that move as the whole policy and a value of +1.
3. Otherwise the side to move searches. With `full_search_share` below 1, only that share of the
   moves gets a full search (`num_mcts_sims_attacker` or `_defender` simulations, with root
   noise), and only those positions are stored. The other moves get a cheap search of
   `cheap_search_sims` simulations, without noise, just to keep the game well played. This is
   KataGo's playout cap randomization: more games per hour, so the value head sees more games.
4. The move is sampled from the visit counts. In `decay` mode the temperature falls from
   `temperature` to 0 over the first `temperature_decay_moves` plies, but never below
   `temperature_floor`.
5. With `reuse_tree`, the subtree of the move just played becomes the root of the next search,
   for both sides. Its visits count toward the next search's simulations.

A game ends by the rules, or as a draw after `max_game_length` plies (the `move_limit` ending).

The search is AlphaZero's PUCT: a child's score is Q + c_puct · P · √N / (1 + n). A child that
has not been visited yet gets the parent's value plus `fpu_reduction`; a negative value makes the
search look deeper into the moves it already likes. `root_fpu_reduction` can set a different
value at the root, where the noise acts. During backup every value is multiplied by
`search_discount` per ply, so of two wins the search prefers the faster one. Root noise is
Dirichlet noise with `dirichlet_alpha`, mixed in with weight `dirichlet_epsilon`.

### What is stored

Every stored position gives:

- the input planes, stored as `uint8` since they hold only 0 and 1;
- the policy target, the share of root visits per move;
- the value target, the game result for the player to move (+1 or −1, and
  `draw_penalty_attacker` or `_defender` for a draw), multiplied by `reward_discount` for every
  ply until the end of the game;
- the ending target, which of `tafl.ENDINGS` the game ended with;
- the moves-left target, the number of plies until the end.

### Replay buffer

The buffer is a ring buffer, so the oldest positions are overwritten first. Its size is counted in
samples, where one position counts as eight: the eight rotations and reflections of the board.
Only one copy is kept, and a random symmetry is applied each time a position is drawn. Training
waits until the buffer holds `min_buffer_size` samples.

### Training

Each iteration trains `num_epochs × batches_per_epoch` batches of `batch_size` positions, drawn
at random from the buffer. The positions of the newest iteration are left out and used for the
validation losses instead, which makes those an honest test on games the network has not trained
on. The optimizer is AdamW with `weight_decay`. The learning rate either decays by `lr_decay` per
iteration down to `lr_floor`, or follows fixed steps (`use_stepping_lr`). Training runs the
network through `torch.compile`.

The loss is the sum of four parts:

- policy: cross-entropy with the visit shares;
- value: squared error, times `value_loss_weight`;
- ending: cross-entropy over the endings the rule set allows, times `ending_loss_weight`;
- moves left: Huber loss in units of 100 plies, times `moves_left_loss_weight`.

Positions can be weighted by how their game ended (`ending_boost`, switched off after
`ending_boost_until`). This helps the network discover a way of winning that self-play reaches
only rarely; switching it off later lets the network settle on what actually works. The weights
are scaled to average 1, so the overall size of the loss stays the same.

### Gating

The new network plays `eval_games` games against the best network so far, with the evaluation
search settings (`eval_c_puct`, `eval_fpu_reduction`) and a temperature for the first
`eval_temperature_threshold` plies. If its Elo gain reaches `eval_min_elo`, it becomes the new
best network and is saved as `best_model.pth`. This runs every `eval_frequency` iterations. With
`eval_interval_growth`, the gap grows in proportion to the iteration number instead, since the
Elo rises fast early and slowly later.

Self-play always uses the newest network, not the best one. So gating only decides which network
is saved as `best_model.pth`, and gives the Elo curve. That curve chains many small steps and
overstates real progress. To compare runs, let their final networks play each other with
`play.py` or `evaluate_elo.py`.

The match against a random player (`eval_vs_random_frequency`) only tells anything very early in
a run, and all configs switch it off.


## Configs

A config has the sections `game`, `network`, `self_play`, `training`, `replay_buffer`,
`evaluation` and `system`. Any key left out takes its default from `TrainingConfig` in
`run_config.py`, where every setting is listed with a comment. The network settings are explained
in `network_config.py`. When a run starts, the resolved config is written to `run_config.json` in
the run directory, so the settings of a run can always be read back.

Every part of the training recipe is a setting, so trying something new usually means writing a
new config rather than changing code. Configs that compare networks keep all other settings equal
to the rule set's `baseline.json`. Configs that test a different recipe say so in their `name`.
`NETWORK_ARCHITECTURES.md` lists the runs and what each one changed.

A few settings change the recipe more than the others:

- `reward_discount` discounts the value target by the plies to the end of the game.
  `search_discount` does the same inside the search only. With `reward_discount` at 1.0 the value
  target is the plain game result.
- `full_search_share` below 1 turns on the cheap searches described above. With fewer stored
  positions per game, `batches_per_epoch` and the buffer size can be smaller for the same amount
  of training per position.
- `lr_decay`, `lr_floor` and `use_stepping_lr` set the learning-rate schedule. `lr_decay` 1.0
  keeps it constant.
- `num_games_per_iteration` and `batches_per_epoch` together set how often each position is
  trained on. Change them together to keep that the same.


## What a run writes

Everything goes to `checkpoints/<rule set>/<run>/`:

- `run_config.json`, the resolved config;
- `output.txt`, the log (`--log`, which `run.sh` sets);
- `training_history.json`, with the losses, timings, self-play statistics and evaluation results
  of every iteration;
- `games.jsonl`, one line per finished game: its type (`selfplay`, `vs_best` or `vs_random`), the
  winner, the ending, the number of plies and the final board;
- `checkpoint_iter_N.pth` every `save_frequency` iterations, `best_model.pth` and, at the end,
  `final_model.pth`. A checkpoint holds the network, the optimizer state, the history and the
  config;
- the plots below, and `tactics_history.jsonl` from `plot_tactics.py`.

To make the plots:

    .venv/bin/python plot.py checkpoints/copenhagen/<run>/training_history.json
    ./plot.sh                                  # every run
    .venv/bin/python plot_games.py checkpoints/copenhagen/<run>/games.jsonl
    .venv/bin/python plot_tactics.py checkpoints/copenhagen/<run>

`plot.py` draws the losses, win rates, endings, Elo, buffer and timing. `plot_games.py` looks at
the final boards: pieces left and game lengths. `plot_tactics.py` runs checkpoints on the
positions in `tactics/` and shows how much of the raw policy goes to the winning move, for each
win condition. It is the quickest way to see whether a network has learned, say, exit forts.


## Playing and analysis

    .venv/bin/python play.py <checkpoint 1> <checkpoint 2> --simulations 400
    .venv/bin/python gui_play.py --checkpoint <checkpoint> --ai-side defender
    .venv/bin/python evaluate_elo.py checkpoints/copenhagen/<run>/ --simulations 400
    .venv/bin/python convert_to_onnx.py checkpoints/copenhagen/<run>/best_model.pth

A checkpoint carries its rule set, network and search settings, so `play.py` and `gui_play.py`
need nothing else. `convert_to_onnx.py` writes a `.onnx` file and, next to it, a `.json` file
with everything a client needs: the input planes in order, the policy layout, the full rule set
and the search settings. The browser game reads both.

The scripts in `analysis/` build opening books from deep searches, compare networks with human
games and build the tactics database; `analysis/README.md` describes them.


## Practical notes

- The workers run one thread each, and `run.sh` gives one worker per physical core. Using all
  hardware threads gives more simulations per second, but the iterations are no faster, because
  more long games are still running when the iteration stops.
- `system.temp_dir` set to `/dev/shm/` keeps the temporary network files in memory.
- Training uses `torch.compile`, which crashes on older CPUs without AVX2. The self-play workers do
  not compile anything, so only the training step is affected.
- A run can be stopped at any time and resumed from its last checkpoint, but the replay buffer
  has to fill up again first.
