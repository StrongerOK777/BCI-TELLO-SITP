# src

`src/` 是公共训练产物目录，由 `TrainUser/train_user.py` 生成和更新。三个硬件的脑控程序默认从这里读取模型。

## 目录结构

```text
src/
  data/      # 脑环训练数据
  models/    # 共享模型
  picture/   # 训练图表
```

## data

保存训练时采集的三类原始数据：

- `actionleft.txt`：左手运动想象
- `actionright.txt`：右手运动想象
- `rest.txt`：静息状态

## models

保存训练得到的模型：

- `FinalModel.pth`：最终共享模型，三个硬件默认读取它
- `model_*_*.pth`：训练过程中的教师模型

## picture

保存训练图表：

- `training_validation_loss.png`
- `model_comparison.png`
- `final_model_training.png`

## 注意

如果重新训练，同名文件会被更新。需要保留旧模型时，建议先备份 `src/models/FinalModel.pth`。
