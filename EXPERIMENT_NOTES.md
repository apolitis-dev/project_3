# Section 6.4: experiment definition

The notebook calls the modified RAPID-LLM simulator in `Rapid-LLM/`, through
`scripts/rapid_experiment.py`. Performance comes from RAPID-LLM's operator
timings and analytical execution graphs. Python creates configurations and
plots the simulator outputs.

## Controlled comparison

- Llama 3.1 8B dimensions, BF16, one simulated A100 with the repository's
  `a100_80GB_no_parallelism.yaml` hardware settings; TP=CP=PP=1, one replica.
- The full-attention control has 32 global layers. The hypothetical variant has
  24 local and 8 global layers, repeating local/local/local/global.
- Windows contain 512, 1,024, 2,048, or 4,096 visible positions **including the
  current token**. Each global layer retains all available positions.
- Prompt lengths: 512, 2,048, 8,192, and 32,768; 256 generated tokens; batch 1.
- Batch comparison: prompt 8,192, window 1,024; batches 1, 2, 4, 8, 16, 24, 32.
- Identical model dimensions, precision, prompt, decode length, and hardware
  within every full/window pair. No weight download or GPU execution occurs.

## Modeling changes

`config.py` parses `attention.sliding_window` and `local_global_interval`.
Interval 4 is local/local/local/global; 0 is all-local; 1 is all-global.

For decode, the existing attention GEMMs and softmax are re-evaluated at
`min(current_positions, window)` for local layers. Query/KV projections,
output projections, FFNs, normalization, and their launch overheads are kept.
The operator timings are combined by local/global layer counts. On a single
serial GPU this gives the same total as executing those two layer types in
sequence; it is explicitly rejected for distributed/MoE configurations.

The resident-cache calculation uses

```
2 * batch * KV_heads * head_dimension * bytes_per_number
  * (global_layers * positions + local_layers * min(positions, window))
```

The graph memory estimator uses the per-layer mean of these cache bytes,
multiplied by the number of resident layers. This is the exact cache total on
the supported single GPU. Existing prompt workspaces remain in peak memory.

**Prefill policy:** dense FlashAttention compute costs are preserved for both
cases, with local masking and eviction assumed. The experiment does not model
an accelerated or chunked local-prefill kernel. It does not measure answer
quality or real cache-management overhead.

Decode sampling includes the window-saturation boundary. A discrete-integration
bug in the upstream sampling code was corrected for both controls and variants:
each interpolated token is now counted exactly once.

`run_perf.py` now writes full-precision `inference_metrics.json`; plots and rate
calculations use this instead of rounded text reports. Infeasible cases are
preserved but omitted from performance curves. Reported batch limits are the
largest **tested** feasible batches, not exhaustive maxima.

## Run and inspect

Run all notebook cells using the project's Jupyter kernel. The simulator runs
in `.rapid-venv`; its analytical dependencies are listed in
`requirements-rapid-analytical.txt`. SCOTCH/AstraSim are not required here.

Each input combination has a stable folder, for example
`runs/rapid/p8192_d256_b32_w1024_g4_s32/`. It holds the latest YAML inputs,
source information, results, simulator reports, and logs. Updated results
replace that folder's contents; historical versions do not accumulate.
The cache key includes runtime source, packages, and configurations. Documentation
changes do not restart simulations.

`runs/main_experiment/results.json` contains the main experiment grid.

Measured validation on a matching trained architecture remains necessary for
real-hardware accuracy claims. The 3:1 layer pattern is a controlled experiment,
not a claim about the trained Llama 3.1 model's architecture or answer quality.

## Published comparisons (Section 7)

`data/published_references.json` records the source URLs, access date, and
reference values. Mistral 7B's published all-local 4,096-position cache at
32,768 positions is reproduced using the paper's dimensions: 8x reduction,
4 GiB versus 0.5 GiB in BF16. The ratio is the published reference; absolute
byte counts are simulator results. The unchanged dense-prefill policy does
not reproduce Mistral's optimized attention/prefill speed claim.

The timing comparison uses all three single-request A100 SXM rows from
Koyeb's Llama 3.1 8B deployment page. Actual input/output counts are used,
not the nominal token-shape labels. Compare published wall-clock seconds
with simulated prefill plus decode seconds, rather than decode throughput.
BF16 is assumed; exact serving precision and kernel versions are unspecified.
No calibration factors were fitted. Absolute errors are approximately
0.04%, 4.60%, and 24.57% (mean 9.74%). This is baseline timing evidence,
not independent validation of the hypothetical mixed-window speedups.
Results overwrite `runs/published_comparison/results.json`.

## Hardware comparisons (Section 8)

`configs/studies/` contains the reference Mistral model and A100-, H100-,
and H200-like hardware profiles. Capacity and bandwidth match NVIDIA's
decimal specifications using numeric bytes: 80 GB / 2.039 TB/s, 80 GB /
3.35 TB/s, and 141 GB / 4.8 TB/s. Compute/cache parameters use repository
models; H200 shares the H100 compute model. These are approximate profiles,
not three independently calibrated physical GPUs. The earlier main
experiment continues to use its original repository hardware configuration.

One matched comparison uses prompt 32,768, decode 256, batch 1, and a
1,024-position window with the 3:1 layer pattern. Controlled bandwidth
variants change only HBM bandwidth (0.5x, 1x, 2x, 4x). Cache byte savings
are identical, while speed improvements vary with the remaining bottlenecks.

Capacity scenarios hold compute and bandwidth fixed and compare RAPID's
predicted peak bytes with 40, 80, and 141 GB thresholds. Batches are
1, 4, 8, 16, 24, 32, 48, 64 at prompt 8,192 and decode 256. The step plots
show the largest member of that finite candidate set that fits; they do not
claim an exhaustive maximum or additional GPU timing measurements. Capacity
alone is not assumed to accelerate a fixed batch.

Hardware/model variants have separate, stable subfolders in `runs/rapid/`.
Repeated results overwrite the same case, and the combined hardware results
overwrite `runs/hardware_study/results.json`. The notebook retains per-run
progress and cached reuse without adding a test framework.
