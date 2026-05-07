import json
import argparse
from trainer import train
import torchvision

def main():
    torchvision.set_image_backend('accimage')  

    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param)

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
