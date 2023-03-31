import configparser


class Config:
    """
    该文件是一个Python模块，主要定义了一个名为Config的类，在初始化时会读取一个配置文件，并将该文件中的参数值解析出来。
    该配置文件中包含了训练和优化器的参数，其中训练参数包括 batch size、最大迭代次数和每次迭代的更新步长；
    优化器参数包括初始学习率、Adam优化器的epsilon值和权重衰减。除此之外，文件还定义了该类的一个__str__方法,用于返回一个包含各个参数值的字符串。
    在文件的最后，还包含了一个主函数，用于测试Config类的功能。
    """
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        # training
        train_config = config['TRAIN']
        self.batch_size = int(train_config['BATCH_SIZE'])
        self.max_steps = int(train_config['MAX_STEPS'])
        self.update_per_step = int(train_config['UPDATE_PER_STEP'])

        # optimizer
        opt_config = config['OPTIMIZER']
        self.init_lr = float(opt_config['INIT_LR'])
        self.adam_eps = float(opt_config['ADAM_EPS'])
        self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

    def __str__(self):
        return 'bs={}_ups={}_lr={}_eps={}_wd={}'.format(
            self.batch_size,
            self.update_per_step,
            self.init_lr,
            self.adam_eps,
            self.adam_weight_decay
        )


if __name__ == '__main__':
    config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
    print(str(Config(config_path)))