"""Run one analytical RAPID-LLM case and preserve its configs and outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


def case_name(model):
    """One stable folder per workload; later results replace earlier results."""
    p = model['model_param']
    attention = p['attention']
    window = attention.get('sliding_window')
    label = 'full' if window is None else f"w{window}_g{attention.get('local_global_interval', 4)}"
    sample = model.get('inference_param', {}).get('sample_every', -1)
    sample = 'auto' if sample == -1 else sample
    return (f"p{p['seq_len']-p['decode_len']}_d{p['decode_len']}_"
            f"b{p['global_batch_size']}_{label}_s{sample}")


def result_fingerprint(provenance, model_text, hardware_text):
    signature = {key: provenance[key] for key in
                 ['commit', 'packages', 'python_version', 'source_sha256']}
    signature['result_schema'] = 2
    return hashlib.sha256((json.dumps(signature, sort_keys=True) + model_text + hardware_text).encode()).hexdigest()


def run_case(repo, output_root, model_path, hardware_path, prompt=None, decode=None, batch=None,
             window=None, interval=4, sample_every=None, hbm_bandwidth=None, hbm_capacity=None):
    repo = Path(repo).resolve()
    model_path = repo / model_path
    hardware_path = repo / hardware_path
    model_text = model_path.read_text()
    hardware_text = hardware_path.read_text()
    model = yaml.safe_load(model_text)
    hardware = yaml.safe_load(hardware_text)
    for field, value in [('bandwidth', hbm_bandwidth), ('size', hbm_capacity)]:
        if value is not None:
            if value <= 0:
                raise ValueError(f'HBM {field} must be positive.')
            hardware['tech_param']['DRAM'][field] = int(value)
    if hbm_bandwidth is not None or hbm_capacity is not None:
        hardware_text = yaml.safe_dump(hardware, sort_keys=False)
    parameters = model['model_param']
    for name, value in [('prompt', prompt), ('decode', decode), ('batch', batch)]:
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive.')
    if hardware['execution_backend']['model'] != 'analytical':
        raise ValueError('This notebook runner uses the analytical backend.')
    if any(hardware['parallelism'][axis] != 1 for axis in ['tp', 'cp', 'pp']):
        raise ValueError('Start with one GPU: tp=cp=pp=1.')
    original_prompt = parameters['seq_len'] - parameters['decode_len']
    if decode is not None:
        parameters['decode_len'] = decode
    if prompt is not None or decode is not None:
        parameters['seq_len'] = (prompt if prompt is not None else original_prompt) + parameters['decode_len']
    if batch is not None:
        parameters['global_batch_size'] = batch
    if window is not None:
        parameters['attention']['sliding_window'] = window
        parameters['attention']['local_global_interval'] = interval
    if sample_every is not None:
        model['inference_param'] = {'sample_every': sample_every}
    if any(value is not None for value in [prompt, decode, batch, window, sample_every]):
        model_text = yaml.safe_dump(model, sort_keys=False)

    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--'], text=True)
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    provenance = {
        'commit': commit,
        'tracked_changes_sha256': hashlib.sha256(changes.encode()).hexdigest(),
        'packages': packages,
        'python_version': sys.version,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(b''.join(
            str(source.relative_to(repo)).encode() + source.read_bytes()
            for source in sorted([*repo.glob('*.py'), *repo.glob('astrasim_lib/*.py')])
        )).hexdigest(),
    }
    # Documentation/test cleanup does not change the simulated workload.
    # Invalidate results for runtime source, environment or input changes only.
    provenance['fingerprint'] = result_fingerprint(provenance, model_text, hardware_text)
    run_id = case_name(model)
    if (model_path.resolve() != (repo / 'configs/model-config/Llama3.1-8B_inf.yaml').resolve()
            or hardware_path.resolve() != (repo / 'configs/hardware-config/a100_80GB_no_parallelism.yaml').resolve()):
        run_id = f'{model_path.stem}__{hardware_path.stem}/{run_id}'
    variants = []
    if hbm_bandwidth is not None:
        variants.append(f'bw{hbm_bandwidth/1e9:g}GBs')
    if hbm_capacity is not None:
        variants.append(f'cap{hbm_capacity/1e9:g}GB')
    if variants:
        parent, name = run_id.rsplit('/', 1) if '/' in run_id else ('variants', run_id)
        run_id = f'{parent}__{"_".join(variants)}/{name}'
    run_dir = Path(output_root).resolve() / run_id
    run_location = os.path.relpath(run_dir, Path(__file__).resolve().parents[1])
    result_path = run_dir / 'results.json'
    provenance_path = run_dir / 'provenance.json'
    if (result_path.exists() and provenance_path.exists()
            and json.loads(provenance_path.read_text()).get('fingerprint') == provenance['fingerprint']):
        return {**json.loads(result_path.read_text()), 'run_dir': run_location, 'cached': True}

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_model = run_dir / 'model.yaml'
    saved_hardware = run_dir / 'hardware.yaml'
    saved_model.write_text(model_text)
    saved_hardware.write_text(hardware_text)
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (run_dir / 'source.patch').write_text(changes)
    if (repo / 'window_attention.py').exists():
        (run_dir / 'window_attention.py').write_bytes((repo / 'window_attention.py').read_bytes())
    command = [
        sys.executable, str(repo / 'run_perf.py'),
        '--hardware_config', str(saved_hardware), '--model_config', str(saved_model),
    ]
    # run_perf replaces output/LLM on each invocation: give every case its own cwd.
    environment = os.environ.copy()
    for key in ['RAPID_MEM_ONLY', 'RAPID_VISUALIZE_GRAPHS', 'RAPID_PERSIST_ASTRASIM_ARTIFACTS', 'RAPID_PERSIST_ARTIFACT_VIZ']:
        environment.pop(key, None)
    try:
        completed = subprocess.run(command, cwd=run_dir, env=environment, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'RAPID-LLM exceeded 300 seconds. Case folder: {run_dir}') from exc
    (run_dir / 'stdout.log').write_text(completed.stdout)
    (run_dir / 'stderr.log').write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'RAPID-LLM failed. See {run_dir}/stderr.log\n{completed.stderr[-2000:]}')

    timing = (run_dir / 'output/LLM/LLM_inference_results.txt').read_text()
    memory = (run_dir / 'output/LLM/memory-summary/memory_capacity_comparison.txt').read_text()

    def number(pattern, text):
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f'Missing result {pattern!r} in {run_dir}')
        return float(match.group(1))

    decode_seconds = number(r'^Decode Time: ([\d.eE+-]+)s', timing)
    metrics_path = run_dir / 'output/LLM/inference_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    decode_seconds = metrics.get('decode_time', decode_seconds)
    if decode_seconds <= 0:
        raise ValueError('Decode duration is below the text report precision; use a longer decode.')
    result = {
        'run_id': run_id,
        'run_dir': run_location,
        'commit': commit,
        'model': model,
        'hardware': hardware,
        'prompt_tokens': parameters['seq_len'] - parameters['decode_len'],
        'decode_tokens': parameters['decode_len'],
        'total_tokens': parameters['seq_len'],
        'batch_size': parameters['global_batch_size'],
        'prefill_s': metrics.get('prefill_time', number(r'^Prefill Time: ([\d.eE+-]+)s', timing)),
        'decode_s': decode_seconds,
        'ttft_s': metrics.get('time_to_first_token', number(r'^Time to First Token: ([\d.eE+-]+)s', timing)),
        'peak_gib': metrics.get('peak_gib', number(r'^Max peak memory usage \(per gpu\): ([\d.eE+-]+)', memory)),
        'capacity_gib': metrics.get('capacity_gib', number(r'^Hardware memory capacity \(per gpu\): ([\d.eE+-]+)', memory)),
        'per_user_tok_s': parameters['decode_len'] / decode_seconds,
        'aggregate_tok_s': parameters['decode_len'] * parameters['global_batch_size'] / decode_seconds,
    }
    if metrics:
        result.update(
            kv_resident_gib=metrics['kv_cache_final_resident_bytes'] / 1024**3,
            kv_resident_bytes=metrics['kv_cache_final_resident_bytes'],
            kv_fetch_bytes=metrics['kv_cache_decode_fetch_bytes'],
            local_layers=metrics['local_layers'], global_layers=metrics['global_layers'],
            window=metrics['sliding_window'], prefill_policy=metrics['prefill_compute_policy'],
        )
    # Use the simulator's capacity decision rather than comparing rounded numbers.
    result['fits'] = '[WARN] Peak memory exceeds capacity' not in memory
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    return {**result, 'cached': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model', default='configs/model-config/Llama3.1-8B_inf.yaml')
    parser.add_argument('--hardware', default='configs/hardware-config/a100_80GB_no_parallelism.yaml')
    parser.add_argument('--prompt', type=int)
    parser.add_argument('--decode', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--window', type=int)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--sample-every', type=int)
    parser.add_argument('--hbm-bandwidth', type=int, help='Override HBM bandwidth in bytes/s.')
    parser.add_argument('--hbm-capacity', type=int, help='Override HBM capacity in bytes.')
    args = parser.parse_args()
    print(json.dumps(run_case(args.repo, args.output_root, args.model, args.hardware,
                             args.prompt, args.decode, args.batch, args.window, args.interval, args.sample_every,
                             args.hbm_bandwidth, args.hbm_capacity)))
