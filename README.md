## 1. Dataset Preparation

- CIFAR-224: The dataset will be downloaded automatically to `./data` on first run.
- ImageNet-A: Place the data under `./data/imagenet-a/train` and `./data/imagenet-a/test`.
- ImageNet-R: Place the data under `./data/imagenet-r/train` and `./data/imagenet-r/test`.
- FGVC-Aircraft: Place the official dataset under `./data/fgvc-aircraft-2013b/data`.

We have implemented the pre-processing datasets as follows:
- CIFAR100: will be automatically downloaded by the code.
- ImageNet-R: https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing
- ImageNet-A: https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing
- FGVC-Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
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

### ImageNet-R
```bash
python main.py --config ./exps/imagenetr.json
```

## 3. Output Location

- Training logs: `./logs/`
