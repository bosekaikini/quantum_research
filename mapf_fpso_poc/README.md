## FPSO-MAPF proof of concept (Firefly-style swarm)

This folder is a **minimal, runnable POC** inspired by the one-pager
`FPSO_Multi_Agent_Pathfinding_One_Pager (1).pdf`, but implemented using the
same **Firefly-style attraction kernel** you already use elsewhere in this repo.

### What it does

- **Grid world** with obstacles (agents live on cell centers in \([0,1]^2\))
- **Greedy baseline** (`baseline.py`) = discrete MAPF with **local greedy** moves (often deadlocks in clutter)
- **A* baseline** (`baseline_astar.py`) = discrete MAPF with **prioritized A* replanning** each step (stronger reference)
- **A* + reserved paths** (`baseline_astar_reserved_path.py`) = prioritized A* where **other agents' trails are impassable**
- **Reserved-path baseline** (`baseline_reserved_path.py`) = same discrete greedy controller, but **cells on another agent's trail become impassable** for everyone else (paths act like dynamic obstacles)
- **Firefly/FPSO (grid)** = continuous swarm with **hard grid** obstacle projection (`sim.py`)
- **Firefly/FPSO (valleys)** = continuous swarm in a **smooth valley field** — agents are pulled into Gaussian wells at obstacle sites (`sim_valley.py`, `valley_env.py`)
- Online update per timestep:
  - **Brightness** increases as distance-to-goal decreases
  - Agents are attracted toward **brighter peers** (Firefly kernel) and toward goal
  - **Collision avoidance** via repulsion and a brightness penalty within radius \(r_\min\)
  - Positions are projected back to free space if they step into obstacles

### Run the demo

From the repo root:

```bash
python -m mapf_fpso_poc.demo
```

The demo prints **baseline vs firefly** summary stats to stdout.

Analysis outputs are saved under `plots/mapf_fpso_poc/` with an auto-incrementing suffix (`_1`, `_2`, …) so reruns do not overwrite previous plots/CSVs.

```bash
python -m mapf_fpso_poc.analysis.run_analysis
```

**Default** comparison (A* discrete grid vs Firefly valley obstacles):

```bash
python -m mapf_fpso_poc.analysis.run_analysis
# -> trajectories_astar_vs_firefly_valleys_N.png
```

Other modes: `--only astar`, `--only astar_reserved`, `--only all`

To skip the baseline run:

```bash
python -m mapf_fpso_poc.demo --no-baseline
```

