## FPSO-MAPF proof of concept (Firefly-style swarm)

This folder is a **minimal, runnable POC** inspired by the one-pager
`FPSO_Multi_Agent_Pathfinding_One_Pager (1).pdf`, but implemented using the
same **Firefly-style attraction kernel** you already use elsewhere in this repo.

### What it does

- **Grid world** with obstacles (agents live on cell centers in \([0,1]^2\))
- **Baseline** = classical-style **discrete MAPF**: one 4-connected move (N/S/E/W/wait) per agent per timestep, with vertex/swap conflict handling
- **Firefly/FPSO** = **continuous** online swarm from the one-pager (velocity + projection); this is the FPSO lift, not a grid solver
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

To skip the baseline run:

```bash
python -m mapf_fpso_poc.demo --no-baseline
```

