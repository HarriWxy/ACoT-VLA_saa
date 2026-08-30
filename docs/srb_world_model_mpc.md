# SRB world-model / CEM-MPC baseline

This baseline turns structured random interaction into a controller without
behavior cloning:

```text
smooth exploration -> complete transitions -> physics-conditioned dynamics
                    -> CEM trajectory search -> execute first action -> replan
```

The new pipeline is intentionally independent of the VLA dataset and
checkpoint.  It only uses low-dimensional state, command, physics and action
arrays.

## 1. Collect transitions

Run this in the SRB/Isaac Sim Python environment, not in the lightweight OpenPI
environment that lacks the `srb` package:

```bash
python scripts/collect_srb_dynamics.py \
  --domains moon,mars,earth \
  --episodes-per-domain 100 \
  --state-keys state,proprio,proprio_dyn,state_dyn \
  --command-keys command \
  --action-repeat 5 \
  --output data/srb_dynamics/transitions.npz
```

If the selected wrapper exposes only `state` and `proprio`, leave the default
state paths in place for a smoke test, but add joint position/velocity and
previous action before using the model for serious locomotion.  The command is
mandatory: a controller cannot optimize velocity tracking if the target command
was not logged.

The collector rebuilds the SRB config with the selected domain before creating
each environment, so its native gravity setup is applied. It also writes native
material friction and, for direct SRB tasks, `action_delay_steps`. Before `gymnasium.make`, it exposes
all five values through `env_cfg.vla_physics` (and direct convenience fields)
for the custom VLA task to consume. Payload mass and motor strength are changed
directly when the task exposes compatible pre-spawn mass or actuator-limit
fields; otherwise the VLA task must apply those two values. Check
`metadata.physics_application.applied` after collection rather than assuming a
requested value affected the simulator.

The saved sidecar JSON contains the state layout, command layout, action bounds,
decision frequency and physics ordering.  Do not change a layout after training
without collecting a new dataset.

## 2. Train the dynamics ensemble

```bash
python scripts/train_srb_world_model.py \
  --data data/srb_dynamics/transitions.npz \
  --output checkpoints/srb_world_model \
  --ensemble-size 5 \
  --batch-size 1024 \
  --epochs 200
```

The script splits by episode, computes state/action/command/physics statistics
on training episodes only, and saves:

```text
checkpoints/srb_world_model/
  model_config.json
  params.msgpack
  stats.json
  metadata.json
```

All four files are required at inference.  Physics normalization is fixed
affine normalization; it is not recomputed per planet or per sample.

## 3. Run CEM-MPC

```bash
python examples/srb/run_cem_mpc.py \
  --checkpoint checkpoints/srb_world_model \
  --domain mars \
  --env-id srb/locomotion_velocity_tracking \
  --num-samples 2048 \
  --num-iterations 5
```

At each decision step the planner samples 2048 correlated action paths of 20
steps, evaluates them with all ensemble members, penalizes predicted falls and
model disagreement, then executes only the first action.  The old plan is
shifted and reused as the next warm start.

## 4. Required acceptance checks

Before using a VLA output as a high-level prior, check:

1. One-step and 10-step rollout error on held-out episodes.
2. Real-environment return for zero, smooth-random and CEM controllers.
3. Correct-physics versus shuffled-physics evaluation.  If shuffling physics
   has no effect, the model is ignoring the physics input.
4. Separate scores for Moon, Mars, Earth and an interpolated gravity value.
5. Fall rate, velocity-tracking RMSE, action-rate cost and energy/torque cost.

The intended final hierarchy is VLA (image/text to velocity command or waypoint)
at a low rate, followed by this physics-aware CEM-MPC controller at 25--50 Hz.
