## 1. Dataset Preparation

- CIFAR-224: The dataset will be downloaded automatically to `./data` on first run.
- ImageNet-A: Place the data under `./data/imagenet-a/train` and `./data/imagenet-a/test`.
- FGVC-Aircraft: Place the official dataset under `./data/fgvc-aircraft-2013b/data`.

## 2. Run Commands

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

## 3. Output Location

- Training logs: `./logs/`
