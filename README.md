# Sliding-Window Attention with RAPID-LLM

Simulates how attention windows affect GPU memory and text-generation speed.
Requires **Git and Python 3.12**. No NVIDIA GPU or model-weight download is needed.

## Setup

After cloning this project, run these commands from its root directory
(macOS/Linux):

    git clone https://github.com/nanocad-lab/Rapid-LLM.git Rapid-LLM
    git -C Rapid-LLM checkout --detach 339ef6ab9fe9b38df3e1d1674ef55121792aa421
    git -C Rapid-LLM apply ../patches/rapid-sliding-window.patch

    python3.12 -m venv .venv
    .venv/bin/python -m pip install jupyterlab matplotlib ipykernel
    .venv/bin/python -m ipykernel install --prefix .venv --name rapid-window --display-name "RAPID window"
    .venv/bin/jupyter lab sliding_window_start.ipynb

The patch adds this project's window and cache model to upstream RAPID-LLM.
The notebook automatically creates .rapid-venv and installs the simulator's
analytical dependencies from requirements-rapid-analytical.txt.

## Run

Select the **RAPID window** kernel, then choose **Restart Kernel and Run All**.
Edit the separate input cells to change prompts, windows, batches, or hardware.
The first run computes the full experiment grid; unchanged cases reuse local
cached results.

Results are saved to:

- runs/main_experiment/results.json
- runs/published_comparison/results.json
- runs/hardware_study/results.json

Reruns overwrite the corresponding results. Detailed per-case output lives in
the ignored runs/rapid/ directory. See [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md)
for assumptions and limitations.

Built on [RAPID-LLM](https://github.com/nanocad-lab/Rapid-LLM), licensed under
Apache 2.0. Its license is included in patches/LICENSE-RAPID-LLM.
