<!-- FOR AI AGENTS - Human readability is a side effect, not a goal -->
<!-- Managed by agent: keep sections and order; edit content, not structure -->
<!-- Last updated: 2026-07-10 | Last verified: 2026-07-10 -->

# AGENTS.md

**Precedence:** the **closest `AGENTS.md`** to the files you're changing wins. This root file holds global defaults.

BCI-TELLO is a Tongji University undergraduate innovation (SITP) project. A NeuroSky / MindWave EEG headset ("脑环", brain ring) is trained once to classify motor-imagery signals, and the shared model drives three hardware targets: a car, a Tello drone, and a DOFBOT robot arm. Human docs live in [README.md](./README.md) (Chinese).

## Commands
> Source: verified by running against this repo — no CI, linter, or type checker is configured.

<!-- AGENTS-GENERATED:START commands -->
| Task | Command | Notes |
|------|---------|-------|
| Test (all) | `python -m unittest discover -s tests` | Runs from repo root; uses stdlib `unittest`, **not** pytest |
| Test (single) | `python -m unittest tests.test_modular_bin.CarBrainControlTests` | Class or `Class.method` |
| Typecheck | — | None configured |
| Lint / Format | — | None configured |
| Build | — | No packaging; scripts run directly with `python` |
<!-- AGENTS-GENERATED:END commands -->

> If a command fails, verify against `MI-DroneControl/requirements.txt` or ask the user before adding tooling.

## Setup & environment
- **Python 3.11.9** is the primary interpreter (`Spaceinvaders/Python2.7ver.` is legacy 2.7 code, not maintained).
- No root `requirements.txt` or `pyproject.toml`. Install deps with pip; the closest manifest is [MI-DroneControl/requirements.txt](./MI-DroneControl/requirements.txt) (`numpy pygame torch pyserial scikit-learn djitellopy matplotlib`). The full library list is in [README.md](./README.md).
- **Headset serial port**: edit [bci_interface.py](./bci_interface.py) only — `MINDWAVE_PORT` (Windows `COM5`, macOS `/dev/cu.usbmodem*`), `MINDWAVE_BAUD` (57600). Every training and brain-control entrypoint reads its default from here; never scatter port config into device scripts.

## Architecture (read before changing behavior)
- **`bin/`** is the reusable core imported by every device app — do not fork logic into device folders:
  - `eeg.py` — `BrainSignalReader`, `EEGSnapshot`, feature/window builders, blink debounce.
  - `models.py` — `FinalUnifiedModel`, `ModelPredictor`, feature-window combining, model loading.
  - `training.py` — dataset + model variants and the training entrypoint.
  - `keyboard.py` — `KeyboardReader` (swappable backend). `transport.py` — `JsonHttpClient`.
- **One shared model**: `TrainUser/train_user.py` trains once and writes `src/models/FinalModel.pth`; all three `brain_control.py` scripts default to that path (override with `--model-path`).
- **Device apps** (`MI-CarControl/`, `MI-DroneControl/`, `MI-DOFBOT/`) each pair a `keyboard_control.py` with a `brain_control.py`. Each `brain_control.py` inserts `ROOT_DIR` into `sys.path` so it can `from bin.eeg import …`. Hardware is isolated behind an adapter (`drone_hardware.py`, `arm_hardware.py`) with a simulated variant (e.g. `SimulatedDroneController`) that the tests exercise.
- **Control logic is a mode state machine**: a double blink toggles modes (e.g. 前后 forward/backward ↔ 转向/转弯 turning ↔ 升降 up/down); attention/meditation drive rule modes, the model drives turning. See `tests/test_modular_bin.py` for the authoritative expected signal sequences.

## File Map
<!-- AGENTS-GENERATED:START filemap -->
```
bci_interface.py       Headset serial config (single source of truth)
bin/                   Reusable core: eeg, models, training, keyboard, transport
TrainUser/             Common training -> src/models/FinalModel.pth
src/                   Training artifacts: data/, models/, picture/
MI-CarControl/         Car: keyboard_control.py, brain_control.py, test_brain.py
MI-DroneControl/       Tello: keyboard/brain control, predict.py, diagnose.py, drone_hardware.py
MI-DOFBOT/             DOFBOT arm: keyboard/brain control, arm_hardware.py, Arm_Lib
Spaceinvaders/         Legacy EEG test game (Python 2.7 + 3.11)
tests/                 unittest suite for bin/ core + controllers
```
<!-- AGENTS-GENERATED:END filemap -->

## Golden Samples (follow these patterns)
| For | Reference | Key patterns |
|-----|-----------|--------------|
| Reusable core module | `bin/eeg.py` | Dataclass results, injectable deps, no device-specific code |
| Device controller + tests | `MI-DroneControl/brain_control.py` + `tests/test_modular_bin.py` | Controller takes reader/predictor/hardware via constructor so a Fake/Simulated impl can be injected |
| Hardware isolation | `MI-DroneControl/drone_hardware.py` | Real controller + `SimulatedDroneController` sharing one interface |

## Heuristics (quick decisions)
| When | Do |
|------|-----|
| Editing control logic | Update/extend `tests/test_modular_bin.py`; assert exact signal/action sequences |
| Adding a signal source or device | Put shared logic in `bin/`, keep only wiring in the device folder |
| Running any `brain_control.py` / `keyboard_control.py` against real hardware | Pass `--dry-run` (and `--test-brain` where available) first |
| Changing headset port | Edit `bci_interface.py`, nothing else |
| Adding a dependency | Ask first — deps are installed by hand, not locked |

## Boundaries

### Always Do
- Run `python -m unittest discover -s tests` after touching `bin/` or any controller, and **show the output** before claiming it passes.
- Keep hardware-independent logic testable: inject reader/predictor/hardware, never construct real serial/network connections at import time.
- Preserve `bci_interface.py` as the single place for headset connection settings.
- Match the surrounding bilingual style: user-facing prints/log strings are Chinese; code identifiers stay English.

### Ask First
- Adding new Python dependencies or introducing packaging/lockfiles.
- Introducing a linter, formatter, type checker, or CI.
- Changing the trained-model format or the default `src/models/FinalModel.pth` path/contract.
- Reviving or restructuring the legacy `Spaceinvaders/Python2.7ver.` code.

### Never Do
- Commit secrets, real headset/device network addresses, or large binaries (`.pth`, `.pptx`, media) that aren't already tracked.
- Test control logic by connecting to a real drone/arm/car — use the simulated controllers and dry-run flags.
- Duplicate `bin/` logic into a device folder to make a quick fix.
- Push directly to `main` — work on a feature branch and open a PR.

## Terminology
| Term | Means |
|------|-------|
| 脑环 / MindWave / NeuroSky | The EEG headset providing the brain signal |
| MI (运动想象) | Motor imagery — the classification paradigm behind `MI-*` folders |
| attention / meditation / poorSignal / blinkStrength | NeuroSky per-sample metrics read by `bin/eeg.py` |
| 前后 / 转向·转弯 / 升降 | Control modes: forward-backward / turning / up-down |
| FinalModel.pth | The one shared trained model consumed by all three devices |
| dry-run | Run control logic without connecting real hardware |

## When instructions conflict
The nearest `AGENTS.md` wins. Explicit user prompts override these files.
