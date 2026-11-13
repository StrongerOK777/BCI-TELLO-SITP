"""Device registry shared by the web panel and the cue test (no Flask needed).

Each entry wires one device folder's ``brain_control.py`` (its config class and
``build_device``) into a factory ``factory(real) -> (settings, device)``; with
``real=False`` the device is the simulated / dry-run variant.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.hub import DeviceOption  # noqa: E402

DEVICE_MODULES = {
    # key: (label, brain_control.py, config class, simulated-config overrides)
    "car": ("小车", "MI-CarControl/brain_control.py", "CarBrainConfig", lambda real: {"dry_run": not real}),
    "drone": ("Tello 无人机", "MI-DroneControl/brain_control.py", "MIDroneConfig", lambda real: {"simulated": not real}),
    "arm": ("DOFBOT 机械臂", "MI-DOFBOT/brain_control.py", "ArmBrainConfig", lambda real: {"simulated": not real}),
}


def load_device_module(key: str):
    _, relative, _, _ = DEVICE_MODULES[key]
    path = ROOT_DIR / relative
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    name = f"{key}_brain_control"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


def device_options(model_path: Optional[str] = None) -> List[DeviceOption]:
    options = []
    for key, (label, _, config_name, sim_overrides) in DEVICE_MODULES.items():
        def factory(real: bool, key=key, config_name=config_name, sim_overrides=sim_overrides) -> Tuple[Any, Any]:
            module = load_device_module(key)
            settings = getattr(module, config_name)(**sim_overrides(real))
            if model_path:
                settings.model_path = model_path
            return settings, module.build_device(settings)

        options.append(DeviceOption(key, label, factory))
    return options


def build_settings(key: str, real: bool, argv: Optional[List[str]] = None) -> Any:
    """Device settings from its own CLI flags (``argv``), forced simulated unless ``real``."""

    module = load_device_module(key)
    settings = module.config_from_args(module.build_arg_parser().parse_args(argv or []))
    for name, value in DEVICE_MODULES[key][3](real).items():
        setattr(settings, name, value)
    return settings
