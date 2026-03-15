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

    # Check TPU
    try:
        import tensorflow as tf
        tpus = tf.config.list_logical_devices('TPU')
        if tpus:
            return f"TPU: {tpus[0].name}"
    except ImportError:
        pass

    try:
        import torch_xla.core.xla_model as xm
        return f"TPU: {xm.xla_device()}"
    except ImportError:
        pass

    return "No GPU or TPU detected (CPU only)"
