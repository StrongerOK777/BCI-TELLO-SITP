<!-- FOR AI AGENTS - Human readability is a side effect, not a goal -->
<!-- Managed by agent: keep sections and order; edit content, not structure -->
<!-- Last updated: 2026-07-11 | Last verified: 2026-07-11 -->

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
  - `models.py` — `FinalUnifiedModel`, `ModelPredictor` (`predict_window` argmax / `predict_proba` softmax), feature-window combining, model loading.
  - `decoding.py` — signal→intent layer: `Decision`, `gate_by_confidence` (softmax rest-band), `VoteWindow` (majority-vote hysteresis). Stdlib-only, device-agnostic.
  - `training.py` — dataset + model variants and the training entrypoint.
  - `keyboard.py` — `KeyboardReader` (swappable backend). `transport.py` — `JsonHttpClient`.
- **One shared model**: `TrainUser/train_user.py` trains once and writes `src/models/FinalModel.pth`; all three `brain_control.py` scripts default to that path (override with `--model-path`).
- **Device apps** (`MI-CarControl/`, `MI-DroneControl/`, `MI-DOFBOT/`) each pair a `keyboard_control.py` with a `brain_control.py`. Each `brain_control.py` inserts `ROOT_DIR` into `sys.path` so it can `from bin.eeg import …`. Hardware is isolated behind an adapter (`drone_hardware.py`, `arm_hardware.py`) with a simulated variant (`SimulatedDroneController`, `SimulatedArmController`) that the tests exercise. All three follow one shape: a `@dataclass *Config`, a controller that takes `reader` / hardware / `predictor` **by constructor**, a `build_controller(...)` factory, and `build_arg_parser` / `config_from_args` / `main`. The arm additionally routes every decision through `bin.decoding` (confidence gate + vote hysteresis) before it moves a servo.
- **Control logic is a mode state machine**: a double blink toggles modes (e.g. 前后 forward/backward ↔ 转向/转弯 turning ↔ 升降 up/down); attention/meditation drive rule modes, the model drives turning. See `tests/test_modular_bin.py` for the authoritative expected signal sequences.

## File Map
<!-- AGENTS-GENERATED:START filemap -->
```
bci_interface.py       Headset serial config (single source of truth)
bin/                   Reusable core: eeg, models, decoding, training, keyboard, transport
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
| Signal→intent decoding | `bin/decoding.py` + `MI-DOFBOT/brain_control.py` | Confidence gate + majority-vote hysteresis; decode kept separate from actuation and unit-tested |

## Code standards & conventions
- **Root cause, minimal diff.** Fix the underlying issue; keep changes surgical and local. Do not reformat or churn unrelated code.
- **Bilingual style (match surroundings).** User-facing prints/logs are Chinese (`模式：`, `执行动作：`, `保持静止`); identifiers, docstrings, and `bin/` comments are English. New device output follows the existing voice.
- **Module boundaries.** Reusable, hardware-independent logic → `bin/` (dataclass results, injected deps, stdlib or torch/numpy only). Device folders hold *wiring only*: a config dataclass, a controller, a hardware adapter, `build_controller`, and the CLI.
- **Dependency injection over globals.** Controllers receive `reader` / hardware / `predictor` through the constructor; `build_controller()` supplies the real defaults. Never open serial/network/model connections at import time — module import must be side-effect-free so tests can load everything without hardware.
- **Paths & config.** `brain_control.py` inserts `ROOT_DIR` (and its own `DEVICE_DIR`) into `sys.path`, then `from bin.… import …`. Headset connection always comes from `bci_interface.get_mindwave_interface()`, never hard-coded per device.
- **Errors degrade to safe.** On bad signal / missing model / prediction failure, print a Chinese warning and fall back to the rest/idle intent (or `safe_stop` / `home`). Never crash the control loop or drive hardware on garbage input.

## Heuristics (quick decisions)
| When | Do |
|------|-----|
| Editing control logic | Update/extend `tests/test_modular_bin.py`; assert exact signal/action sequences |
| Adding a signal source or device | Put shared logic in `bin/`, keep only wiring in the device folder |
| Running any `brain_control.py` / `keyboard_control.py` against real hardware | Pass `--dry-run` (and `--test-brain` where available) first |
| Changing headset port | Edit `bci_interface.py`, nothing else |
| Adding a dependency | Ask first — deps are installed by hand, not locked |

## Testing & verification
- **Framework:** stdlib `unittest`, one suite in `tests/test_modular_bin.py`. No pytest, no network, no real hardware.
- **Assert invariants, not internals.** Feed a list of `WindowResult` / `FeatureWindowResult` into a controller built from `FakeReader` / `FakePredictor` / `FakeSender` + a `Simulated*Controller`, then assert the exact recorded `actions` / `signals` sequence. `bin/` primitives (`VoteWindow`, `gate_by_confidence`, feature builders) get direct unit tests.
- **Ship logic changes with a test** that pins the new sequence; every device controller and every `bin/` primitive already has coverage to extend.
- **Run + show:** `python -m unittest discover -s tests` from the repo root after touching `bin/` or any controller, and paste the `Ran N tests … OK` line before claiming green.

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

## Documentation sync
- Changed CLI flags, control logic, or the mode machine → update the device `readme.md`/`README.md` **and** the matching section of the root [README.md](./README.md).
- Added/renamed a `bin/` module or a shared contract → update the *Architecture* + *File Map* here and the README repo-structure block.
- Describe the **current** state; delete obsolete narrative instead of appending "changed to…". When you edit this file, keep the section order and bump the `Last updated` / `Last verified` header dates.

## Git & release workflow
- Work lands on `dev`; branch off it for a feature and open a PR **into `dev`**. `main` is the release branch — never push to it directly.
- Keep commits focused and match the existing bilingual style (`update:…`, `docs:…`, `fix:…`). Do not commit `.pth`, `.pptx`, media, secrets, or real device/network addresses.

## Terminology
| Term | Means |
|------|-------|
| 脑环 / MindWave / NeuroSky | The EEG headset providing the brain signal |
| MI (运动想象) | Motor imagery — the classification paradigm behind `MI-*` folders |
| attention / meditation / poorSignal / blinkStrength | NeuroSky per-sample metrics read by `bin/eeg.py` |
| 前后 / 转向·转弯 / 升降 | Control modes: forward-backward / turning / up-down |
| FinalModel.pth | The one shared trained model consumed by all three devices |
| intent / 意图 | A decoded command (e.g. `base_left`, `rest`) emitted by `bin/decoding` before actuation |
| confidence gate / vote window | Softmax rest-band + majority-vote hysteresis that stabilize noisy per-window decisions |
| dry-run | Run control logic without connecting real hardware |

## When instructions conflict
The nearest `AGENTS.md` wins. Explicit user prompts override these files.
