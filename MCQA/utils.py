def get_accelerator():
    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return f"GPU: {result.stdout.strip()}"
    except FileNotFoundError:
        pass

    # Check TPU via JAX (works without GCE metadata server)
    try:
        import jax
        tpu_devices = jax.devices('tpu')
        if tpu_devices:
            # e.g. TpuDevice coords tell you the topology
            return f"TPU: {len(tpu_devices)}x {tpu_devices[0].device_kind}"
    except (ImportError, RuntimeError):
        pass

    # Fallback: torch_xla
    try:
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices('TPU')
        if devices:
            return f"TPU: {len(devices)} devices via torch_xla"
    except (ImportError, RuntimeError):
        pass

    return "No GPU or TPU detected (CPU only)"
