# -- coding:UTF-8 --

from config import Config
from train import train_start
from test import predict_start, test_start
from stu_train import stu_train_start
if __name__ == '__main__':
    config = Config()

    # train
    # train_start(config)

    # test_evaluate
    # test_start(config)
    # output result
    # predict_start(config)

    # bert distillation
    stu_train_start(config)
