import os
import numpy as np
import pandas as pd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torch import nn, tensor, Tensor
from tqdm import tqdm


def makeTrainTest(csv_path, target_path=".", pn_factor=4, test_factor=0.3, val_factor=0.1):

    df = pd.read_csv(csv_path)
    total = len(df)
    counts = df["failure"].value_counts()
    num_failure = counts[1]
    factor_failure = num_failure / total
    num_test = total * test_factor
    num_val = total * val_factor
    num_train = total - num_test - num_val

    num_train_f = int(num_train * factor_failure)
    num_train_expected = num_train_f * pn_factor
    num_train_g = min(num_train - num_train_expected, num_train_expected)

    num_test_f = int(num_test * factor_failure)
    num_test_expected = num_test_f * pn_factor
    num_test_g = min(num_test - num_test_expected, num_test_expected)

    num_val_f = int(num_val * factor_failure)
    num_val_expected = num_val_f * pn_factor
    num_val_g = min(num_val - num_val_expected, num_val_expected)

    print("Train samples: {} good disks and {} failures, total: {}"
          .format(num_train_g, num_train_f, num_train_g + num_train_f))
    print("Test samples: {} good disks and {} failures, total: {}"
          .format(num_test_g, num_test_f, num_test_g + num_test_f))
    print("Validation samples: {} good disks and {} failures, total: {}"
          .format(num_val_g, num_val_f, num_val_g + num_val_f))

    good_df = df[df["failure"] == 0]
    failure_df = df[df["failure"] == 1]

    trainval_df_g = good_df.sample(num_train_g + num_val_g)
    test_df_g = good_df[~good_df.index.isin(trainval_df_g.index)].sample(num_test_g)
    train_df_g = trainval_df_g.sample(num_train_g)
    val_df_g = trainval_df_g[~trainval_df_g.index.isin(train_df_g.index)]

    print("trian good: ", len(train_df_g))
    print("test good: ", len(test_df_g))
    print("val good: ", len(val_df_g))

    trainval_df_f = failure_df.sample(num_train_f + num_val_f)
    test_df_f = failure_df[~failure_df.index.isin(trainval_df_f.index)].sample(num_test_f)
    train_df_f = trainval_df_f.sample(num_train_f)
    val_df_f = trainval_df_f[~trainval_df_f.index.isin(train_df_f.index)]

    print("train bad: ", len(train_df_f))
    print("test bad: ", len(test_df_f))
    print("val bad: ", len(val_df_f))

    train_df = pd.concat([train_df_g, train_df_f])
    test_df = pd.concat([test_df_g, test_df_f])
    val_df = pd.concat([val_df_g, val_df_f])

    print("Train sample: ", len(train_df))
    print("Val sample: ", len(val_df))
    print("Test sample: ", len(test_df))

    train_df.to_csv(os.path.join(target_path, "train_sample.csv"), index=False)
    test_df.to_csv(os.path.join(target_path, "test_sample.csv"), index=False)
    val_df.to_csv(os.path.join(target_path, "val_sample.csv"), index=False)


class ST4Dataset(Dataset):
    def __init__(self, csv_path: str, log_path: str):
        super(ST4Dataset, self).__init__()
        self.csv_path = csv_path
        self.log_path = log_path
        self.csv_file = open(csv_path, "r")
        self.csv_lines = self.csv_file.readlines()[1:]
        self.num_lines = len(self.csv_lines) - 1

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        entry = self.csv_lines[index].strip().split(",")
        serial, failure = entry[0], int(entry[1])
        disk_df = pd.read_csv(os.path.join(self.log_path, "{}.csv".format(serial)))
        disk_df.sort_values("date", inplace=True)
        if failure == 1:
            disk_df = disk_df.drop(disk_df[disk_df["failure"] == 1].index)
        disk_df = disk_df.drop(columns=["date", "failure"])
        len_df = len(disk_df)
        disk_np = np.array(disk_df)

        disk_ts = tensor(disk_np[-min(30, len_df):]) # N * 12
        # disk_ts = tensor(disk_np)
        disk_ts = torch.nn.functional.normalize(disk_ts, dim=1)

        # reg:
        if failure == 0:
            tgt = tensor([1, 0])
        else:
            tgt = tensor([0, 1])
        # cls:
        # tgt = tensor([failure])
        return disk_ts, tgt


if __name__ == "__main__":
    makeTrainTest("./data/2015/hasFailure.csv", target_path="./data/2015/", pn_factor=2)

    # st4 = ST4Dataset("./data/train_sample.csv", "./data/ST4000/preprocessed/")
    # target, label, _ = st4[2]
    # print(len(st4))
    # print(target)
    # print(label)
    # print(_)
    #
    # for stage in ["train", "test", "val"]:
    #     print("In {}".format(stage))
    #     path = "./data/{}_sample.csv".format(stage)
    #     st4 = ST4Dataset(path, "./data/ST4000/preprocessed/")
        # for x, y, _ in st4:
        #     if x.shape[0] != 30:
        #         print(_)

