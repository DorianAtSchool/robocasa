# Model Backends

Backends adapt a specific VLA policy family to the shared RoboCasa evaluation
workflow in `model_evals/common/`, `model_evals/scripts/`, and
`model_evals/analysis/`.

Current backends:

```text
gwp/           # GigaWorld-Policy backend source and server/client scripts
pi05/          # OpenPI pi0.5 backend
rldx/          # RLDX-1 backend
gr00t_n1_5/    # GR00T N1.5 backend
```

Each backend has its own `README.md` with environment setup, checkpoint
download, server launch, and RoboCasa client commands.

The goal is for each backend to expose a comparable client entrypoint so the
same one-robot vs. two-robot evaluation scripts and analysis tools can be reused
across models.

Backend code should stay limited to model-specific checkpoint loading,
preprocessing, inference, and action postprocessing. RoboCasa rollout behavior,
trajectory/fallback robot spawning, camera recording, and stats formatting live
in `model_evals/common/`.
