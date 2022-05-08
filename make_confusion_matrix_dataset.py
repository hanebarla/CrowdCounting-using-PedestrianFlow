import random
import csv
from progress.bar import Bar
import argparse
import os


def csv_file_concat(file_list, file_name):
    file_data_list = []
    for file in file_list:
        with open(file, mode="r") as f:
            reader = csv.reader(f)
            for row in reader:
                file_data_list.append(row)
        # file_data_list.append(pd.read_csv(file).reset_index(drop=True))
    # df = pd.concat(file_data_list, axis=0, sort=False)
    # df.to_csv(file_name)
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(file_data_list)


def cross_dataset(args, train_list, val_list, test_list, concat_file_index):
    train_file_list = []
    for file_name in train_list:
        train_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}.csv'.format(file_name)))
        train_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_hDyn.csv'.format(file_name)))
    csv_file_concat(train_file_list, os.path.join(args.savefolder, '{}_train.csv'.format(concat_file_index)))

    val_file_list = []
    for file_name in val_list:
        val_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}.csv'.format(file_name)))
        val_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_hDyn.csv'.format(file_name)))
    csv_file_concat(val_file_list, os.path.join(args.savefolder, '{}_val.csv'.format(concat_file_index)))

    test_file_list = []
    for file_name in test_list:
        test_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}.csv'.format(file_name)))
        test_file_list.append(os.path.join(args.savefolder, 'Scene_IM0{}_hDyn.csv'.format(file_name)))
    csv_file_concat(test_file_list, os.path.join(args.savefolder, '{}_test.csv'.format(concat_file_index)))


def main(args):
    A_train_dataset = [1, 2, 3]
    A_val_dataset = [4]
    A_test_dataset = [5]
    cross_dataset(args, A_train_dataset, A_val_dataset, A_test_dataset, 'A')

    B_train_dataset = [2, 3, 4]
    B_val_dataset = [5]
    B_test_dataset = [1]
    cross_dataset(args, B_train_dataset, B_val_dataset, B_test_dataset, 'B')

    C_train_dataset = [3, 4, 5]
    C_val_dataset = [1]
    C_test_dataset = [2]
    cross_dataset(args, C_train_dataset, C_val_dataset, C_test_dataset, 'C')

    D_train_dataset = [4, 5, 1]
    D_val_dataset = [2]
    D_test_dataset = [3]
    cross_dataset(args, D_train_dataset, D_val_dataset, D_test_dataset, 'D')

    E_train_dataset = [5, 1, 2]
    E_val_dataset = [3]
    E_test_dataset = [4]
    cross_dataset(args, E_train_dataset, E_val_dataset, E_test_dataset, 'E')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('savefolder', help='path to csv folder')
    args = parser.parse_args()

    main(args=args)
