from torch.utils.data import Dataset

from dataset.database import get_database_split, parse_database_name


class DummyDataset(Dataset):
    default_cfg = {
        'database_name': '',
    }

    def __init__(self, cfg, is_train, dataset_dir=None): # 初始化数据集调用
        self.cfg = {**self.default_cfg, **cfg} # 更新数据集的配置；
        if not is_train:
            database = parse_database_name(self.cfg['database_name'], dataset_dir) # database_name表示是哪类数据，dtaset_dir表示数据所在的路径；针对不同的数据使用不同的数据解析函数
            train_ids, test_ids = get_database_split(database, 'validation') #分割训练集和测试集
            self.train_num = len(train_ids) # 训练数据集的数量为总数据量-1
            self.test_num = len(test_ids) # 1
        self.is_train = is_train

    def __getitem__(self, index):  # 索引/迭代时调用
        if self.is_train:
            return {}
        else:
            return {'index': index}

    def __len__(self):
        if self.is_train:
            return 99999999 #?为什么返回的都是这么大的数字
        else:
            return self.test_num
