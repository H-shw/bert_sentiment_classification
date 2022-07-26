# -- coding:UTF-8 --
# config file

class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        # device
        self.device = 'cuda:1'

        # bert
        self.bert_path = '/mnt/sdb/suhangwu/BERT'
        self.vocab_path = '/mnt/sdb/suhangwu/BERT/vocab.txt'
        self.hidden_size = 768
        self.class_num = 3


        # train
        self.epoch = 5
        self.batch_size = 30
        self.shuffle = True
        self.reader_num = 10
        self.learning_rate = 1e-5
        self.step_size = 1
        self.lr_multiplier = 1
        self.train_path = 'dataset/res/div/train.csv'
        self.valid_path = 'dataset/res/div/valid.csv'
        self.test_path = 'dataset/res/div/test.csv'
        self.weight_decay = 0.01
        self.lr = 1e-5
        self.eps = 1e-6
        self.correct_bias = False

        # save
        self.save_path = 'res/stu_model.pt'
        self.res_path = 'res/ans.csv'

        # test
        self.test = 'dataset/res/fin/test.csv'

        # student bert model
        self.stu_vocab_size = 21128
        self.stu_hidden_size = 600
        self.tea_hidden_size = 768
        self.stu_hidden_layers = 6
        self.tea_hidden_size = 768
        self.softloss_factor = 0.2
        self.temperature = 0.8


