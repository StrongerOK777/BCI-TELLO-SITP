# NiceRice with DOFBOT-SE + NeuroSky

> Tongji University SITP project notes and code for controlling a DOFBOT-SE robotic arm with keyboard input and NeuroSky / MindWave EEG signals.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Hardware](https://img.shields.io/badge/Hardware-DOFBOT--SE-orange)
![EEG](https://img.shields.io/badge/EEG-NeuroSky%20MindWave-6f42c1)
![Status](https://img.shields.io/badge/Status-Prototype%20%26%20SITP-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository records the porting, testing, and development work for a DOFBOT-SE robotic arm and NeuroSky EEG headset. The current codebase contains two practical control paths:

- Keyboard control for direct manual testing of the robotic arm.
- Brain signal control for using NeuroSky attention, meditation, and blink signals to drive safe arm actions.

The goal is not a flashy demo. The goal is a stable, adjustable, and safe foundation for gradually improving EEG-based robotic arm control.

## Repository Map

```text
NiceRice-with-DOFBOT-SE-NEUROSKY/
├── brain_arm_control/
│   ├── brain_arm_control.py   # EEG window sampling + robotic arm control
│   └── README.md              # Brain-control usage guide
├── keyboard_control/
│   ├── keyboard_control.py    # Real-time keyboard control program
│   └── README.md              # Keyboard-control usage guide
├── Arm_Lib (Windows)/         # DOFBOT-SE arm control library
├── Arm_Lib.egg-info/          # Python package metadata
├── LICENSE
├── readme.md                  # This project homepage
└── setup.py
```

## Main Modules

| Folder | Purpose | Best For |
|---|---|---|
| `brain_arm_control/` | Reads NeuroSky EEG signals and maps them to robotic arm actions with safety checks. | EEG control experiments and dry-run testing. |
| `keyboard_control/` | Controls the arm with `W/A/S/D`, arrow keys, `Q/E`, `Space/X`, and reset. | Verifying servo direction, wiring, and arm behavior. |
| `Arm_Lib (Windows)/` | Low-level serial wrapper used by the control programs. | Sending servo commands through the official arm API. |

## Quick Start

Clone the repository:

```bash
git clone https://github.com/907nicerice/NiceRice-with-DOFBOT-SE-NEUROSKY.git
cd NiceRice-with-DOFBOT-SE-NEUROSKY
```

Install basic dependencies:

```bash
pip install pyserial pynput
```

For EEG model prediction, PyTorch is optional and only needed when you pass `--model-path` to the brain-control program.

## Brain-Control Program

Go to the EEG control folder:

```bash
cd brain_arm_control
```

Dry-run first. This reads the headset but does not move the arm:

```bash
python brain_arm_control.py --mindwave-port COM6 --dry-run
```

Run with the real arm:

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4
```

Run with an optional model:

```bash
python brain_arm_control.py --mindwave-port COM6 --arm-port COM4 --model-path FinalModel.pth
```

Core behavior:

- `attention` higher than `meditation` drives one direction.
- `meditation` higher than `attention` drives the opposite direction.
- Single blink toggles the gripper.
- Double blink switches mode.
- Poor signal blocks motion; signal loss returns the arm to a safe pose.

Read the full guide: [`brain_arm_control/README.md`](brain_arm_control/README.md)

## Keyboard-Control Program

Go to the keyboard control folder:

```bash
cd keyboard_control
```

Run the control script:

```bash
python keyboard_control.py
```

Default mapping:

| Input | Servo | Action |
|---|---:|---|
| `A` / `D` | 1 | Base left / right |
| `W` / `S` | 2 | Forward / backward |
| `↑` / `↓` | 3 | Up / down |
| `Q` / `E` | 4 | Auxiliary joint decrease / increase |
| `Space` / `X` | 6 | Gripper or end-effector open / close direction |
| `R` | all | Reset to safe middle pose |

Read the full guide: [`keyboard_control/README.md`](keyboard_control/README.md)

## Safety Notes

- Test with `--dry-run` before connecting the robotic arm.
- Keep `angle_step` small. The brain-control program clamps it to at most 10 degrees per action.
- The safe home pose is `[90, 90, 90, 90, 135, 90]`.
- Do not bypass `Arm_Lib.py` to manually write serial frames unless you are debugging the low-level driver.
- Stop immediately if the arm moves in an unexpected direction, then verify the mapping with `keyboard_control.py`.

## Development Roadmap

- Improve EEG threshold calibration for different users.
- Add more robust blink classification.
- Add optional recorded EEG sessions for offline tuning.
- Improve model integration and document expected checkpoint format.
- Add short demo videos and wiring diagrams.

## Credits

Maintained by `907nicerice` for the Tongji University SITP project.

This project is part experiment log, part robotic engineering notebook, and part playground for turning brain-computer interface ideas into careful, testable control software.
