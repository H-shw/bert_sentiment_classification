# -- coding:UTF-8 --

from config import Config
from train import train_start
from test import predict_start, test_start

if __name__ == '__main__':
    config = Config()

    # train
    train_start(config)

    # test_evaluate
    # test_start(config)
    # output result
    # predict_start(config)
