# 口令测试（Online Cue Test）

屏幕依次随机提示 **左转 / 右转 / 静止**，被试用脑环完成口令。脚本记录每条口令的命中情况和耗时，并生成可以直接放进答辩材料的统计表。

被测的是设备 `brain_control.py` 里**真实的**转弯解码流程（`FocusSteer`、机械臂的多窗投票、可选的模型），硬件换成只记录指令的模拟版本。测试期间小车、无人机、机械臂都**不会真的动**；真机演示请另外进行。

## 运行

```bash
# 无人机转弯解码，每类 10 条，共 30 条（约 6–8 分钟）
python experiments/online_cue_test/cue_test.py --device drone --subject S01

# 机械臂，同时调整专注阈值（参数会透传给 MI-DOFBOT/brain_control.py）
python experiments/online_cue_test/cue_test.py --device arm --subject S01 --focus-threshold 60

# 用共享模型做转弯（对比规则方式）
python experiments/online_cue_test/cue_test.py --device car --subject S01 --turning-source model

# 没有脑环时先试跑流程（模拟信号，结果不代表真实性能）
python experiments/online_cue_test/cue_test.py --device car --mock-headset
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | 必填 | `car` / `drone` / `arm` |
| `--trials-per-class` | 10 | 每类口令次数，顺序随机、同一口令最多连续 2 次 |
| `--timeout` | 15 | 左转 / 右转口令的最长等待秒数（约 5 个 3 秒窗口） |
| `--rest-hold` | 9 | 静止口令需要保持不动的秒数 |
| `--gap-sec` | 3 | 两条口令之间的放松准备时间 |
| `--subject` | anon | 被试代号，写进结果目录名（**不要用真实姓名**） |
| `--seed` | 按时间 | 口令顺序随机种子，固定后可复现同一顺序 |
| `--no-gui` | 关 | 不弹出 pygame 窗口，只在终端提示 |
| `--mock-headset` | 关 | 用脚本模拟脑环信号，配合 `--mock-accuracy` |

其他参数（`--focus-threshold`、`--turning-source`、`--model-path`、`--mindwave-port` 等）原样传给对应设备的 `brain_control.py`。窗口里按 `ESC` 或关闭窗口会提前结束，已完成的口令照常出报告。

## 判定规则

| 口令 | 命中 | 失败 |
|---|---|---|
| 左转 / 右转 | 在 `--timeout` 内发出的**第一个**转向动作与口令一致 | 方向错误；超时未动作 |
| 静止 | `--rest-hold` 内没有任何转向动作 | 误动作 |

- **耗时**：从口令出现到第一个转向动作。规则模式下，如果当前方向不对，需要先单眨眼切换方向，所以耗时包含这一步；屏幕会提示"当前方向：右（先单眨眼切换）"。
- **双眨眼**：会被控制器当作切换模式。测试中会自动切回转弯模式，并在终端提示。
- **机械臂**：每条口令开始前会清空多窗投票，因为口令间隔里被试处于放松状态。

## 输出

结果写在 `results/<设备>_<被试>_<时间>/`：

- `summary.md`：答辩用表格，包括：
  - 每类口令的命中率、平均 / 中位耗时、失败原因；
  - 总体命中率与显著性阈值（`bin.training.significance_threshold`，p<0.05）；
  - 平均响应时间、Wolpaw 信息传输率（ITR）、无效窗口数；
  - 口令 × 实际动作的混淆矩阵。
- `trials.csv`：逐条记录（口令、结果、实际动作、耗时、窗口数、无效窗口、口令出现时的当前方向）。
- `summary.json`：上述统计和本次使用的参数（专注阈值、转弯信号来源、窗口大小等）。

## 建议的测试流程

1. 先运行 `python MI-CarControl/brain_control.py --test-brain`，分别保持专注和放松，看 attention 中位数，选好 `--focus-threshold`。
2. 每位被试先做 1 轮练习（`--trials-per-class 3`），再做正式的 30 条。
3. 至少测 2–3 位被试。条件允许的话，规则方式和 `--turning-source model` 各测一轮，对比两者的命中率和 ITR。
