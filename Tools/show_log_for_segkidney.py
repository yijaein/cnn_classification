import argparse
import os

import matplotlib.pyplot as plt


def norm_path(path):
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    path = os.path.normcase(path)
    path = os.path.abspath(path)
    return path


def parseCVS(log_file, format=None):
    buf = []
    with open(log_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            split_line = line.strip().split('\t')

            if format is not None:
                for idx, dataType in enumerate(format):
                    item = split_line[idx]

                    if dataType == 'int':
                        item = int(item)
                    elif dataType == 'float':
                        item = float(item)
                    elif dataType == 'str' or dataType == None:
                        pass
                    split_line[idx] = item
            buf.append(split_line)

    return buf


def main(args):
    log_file = norm_path(args.log)

    train_log = os.path.join(log_file, 'log_train.txt')
    val_log = os.path.join(log_file, 'log_valid.txt')

    train_buf = parseCVS(train_log, format=('str', 'int', 'float', 'float'))
    val_buf = parseCVS(val_log, format=('str', 'int', 'float', 'float'))

    train_loss_y = [item[2] for item in train_buf]
    train_loss_x = range(1, len(train_loss_y)+1)
    val_loss_y = [item[2] for item in val_buf]
    val_loss_x = range(1, len(val_loss_y)+1)

    train_psnr_y = [item[3] for item in train_buf]
    train_psnr_x = range(1, len(train_psnr_y)+1)
    val_psnr_y = [item[3] for item in val_buf]
    val_psnr_x = range(1, len(val_psnr_y)+1)


    fig = plt.figure()
    ax = plt.subplot(121)
    ax.plot(train_loss_x, train_loss_y, label='train_loss')
    ax.plot(val_loss_x, val_loss_y, label='valid_loss')
    ax.legend()

    ax = plt.subplot(122)
    ax.plot(train_psnr_x, train_psnr_y, label='train_psnr')
    ax.plot(val_psnr_x, val_psnr_y, label='valid_psnr')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show log graph')
    parser.add_argument('--log', default='../SegKidney/checkpoint', help='path to log')
    args = parser.parse_args()

    main(args)
