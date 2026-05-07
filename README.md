# Incremental Learning Project

一个基于 ViT 多适配器的增量学习训练代码仓库，支持通过配置文件快速启动实验。

## 1. 环境准备

建议使用 Python 3.9+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm numpy scipy pillow tqdm matplotlib scikit-learn
```

## 2. 数据准备

- CIFAR-224：首次运行会自动下载到 `./data`。
- ImageNet-A：请将数据放到 `./data/imagenet-a/train` 和 `./data/imagenet-a/test`。
- FGVC-Aircraft：请将官方数据放到 `./data/fgvc-aircraft-2013b/data`。

## 3. 运行命令

### CIFAR-224
```bash
python main.py --config ./exps/cifar.json
```

### Aircraft
```bash
python main.py --config ./exps/aircraft.json
```

### ImageNet-A
```bash
python main.py --config ./exps/imageneta.json
```

## 4. 输出位置

- 训练日志：`./logs/`
- 模型保存路径：由配置文件中的 `model_dir` 指定（如 `save_model/cifar224/`）。
