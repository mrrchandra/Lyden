#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/Acer/Desktop/Projs/Lyden
/home/madrin/venvs/lyden/bin/python - <<'EOF'
import inspect, traceback

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    print("--- Mamba2.forward signature ---")
    print(inspect.signature(Mamba2.forward))
    sig_params = inspect.signature(Mamba2.__init__).parameters
    print("--- Mamba2.__init__ params ---")
    for name in sig_params:
        print(" ", name)
    print("--- default values of interest ---")
    for k in ("d_model", "d_state", "d_conv", "expand", "headdim", "chunk_size"):
        if k in sig_params:
            print(f"  {k} = {sig_params[k].default!r}")
    try:
        from mamba_ssm.modules.mamba_simple import InferenceParams  # where it lives in Mamba1
        print("InferenceParams: mamba_simple")
    except ImportError:
        import mamba_ssm.utils.generation as gen
        print("InferenceParams attrs:", [m for m in dir(gen.InferenceParams) if not m.startswith("_")][:12])
    import mamba_ssm.utils.generation as gen
    ip = getattr(gen, "InferenceParams", None)
    if ip is None:
        try:
            from mamba_ssm.modules.mamba_simple import InferenceParams as ip  # type: ignore
        except ImportError as e:
            print("no InferenceParams:", e)
    print("fields:", ip.__init__.__doc__ or "")
except Exception:
    traceback.print_exc()
EOF
