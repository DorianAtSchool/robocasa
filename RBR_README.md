Set up a local `.venv` and the in-repo `robosuite` dependency before running the examples:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
git submodule update --init --recursive
uv pip install -e ./robosuite
uv pip install -e .
```

If you are cloning the repo for the first time, you can also use `git clone --recurse-submodules ...` to fetch `robosuite/` immediately.

Run random policy rollout:

```bash
python policies/random.py
```

This writes a rollout video to `test.mp4` at the repo root.
