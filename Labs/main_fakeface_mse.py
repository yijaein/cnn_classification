import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from Official.densenet import _DenseBlock, _Transition
from Official.main import main
from Official.utils import AverageMeter, accuracy
from Works.utils import compute_auroc, softmax, save_auroc

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_IDS = [0,1]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='~/data/fakeface4', help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=0.00001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', type=int, default=30, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--result', default='../result_fakeface_mse', help='path to result')
parser.add_argument('--resize_image_width', type=int, default=160, help='image width')
parser.add_argument('--resize_image_height', type=int, default=256, help='image height')
parser.add_argument('--image_width', type=int, default=160, help='image crop width')
parser.add_argument('--image_height', type=int, default=256, help='image crop height')
parser.add_argument('--avg_pooling_width', type=int, default=10, help='average pooling width')
parser.add_argument('--avg_pooling_height', type=int, default=16, help='average pooling height')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--seg_data', metavar='DIR', default='~/data/fakeface4_seg', help='dataset path for segmentation')
parser.add_argument('--do_classify', action='store_true', default=False, help='do classification')
parser.add_argument('--do_seg', action='store_true', default=True, help='do segmentation')
parser.add_argument('--acc_classify', action='store_true', default=False, help='True: accuracy, False: Seg accuracy')
parser.add_argument('--train_per_loss', action='store_true', default=False, help='True: accuracy, False: Seg accuracy')
parser.add_argument('--target_index', default=1, type=int, help='target index')
args = parser.parse_args()

args.seg_data = os.path.expanduser(args.seg_data)

# Think better way to handle this MSE
criterionMSE = nn.MSELoss().cuda()

# Think better way to handel seg_pil_loader
def seg_pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L')
            img = img.resize((args.avg_pooling_width, args.avg_pooling_height))
            return img

class MultiTaskDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.seg_loader = seg_pil_loader
        self.transform = transform
        self.target_transform = target_transform
        self.seg_transform = transforms.Compose([transforms.ToTensor()])

        black_image_path = 'black.png'
        resize_image_size = (args.resize_image_width, args.resize_image_height)
        black = Image.new('L', resize_image_size, "black")
        black.save(black_image_path)

        # read folder and make dataset
        self.samples = []  # (input, target, seg_image_path)

        list_dir = os.listdir(self.root)
        list_dir.sort()
        for label, sub_dir in enumerate(list_dir):
            sub_dir = os.path.join(self.root, sub_dir)

            for (path, dir, files) in os.walk(sub_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == '.png' or ext == '.jpg':
                        image_path = os.path.join(path, filename)
                        if filename[0] != 'n':
                            seg_image_path = os.path.join(args.seg_data, filename)
                        else:
                            seg_image_path = black_image_path

                        self.samples.append((image_path, label, seg_image_path))

    def __getitem__(self, index):
        path, target, seg_path = self.samples[index]
        sample = self.loader(path)
        seg_sample = self.seg_loader(seg_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.seg_transform is not None:
            seg_sample = self.seg_transform(seg_sample)

        return sample, target, seg_sample

    def __len__(self):
        return len(self.samples)


class densenet_multi(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, channels=3, num_classes=1000, avg_pooling_size=7):

        super(densenet_multi, self).__init__()

        self.avg_pooling_size = avg_pooling_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.seg_classifier = nn.Conv2d(num_features, 1, kernel_size=1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features, inplace=True)

        out = F.max_pool2d(features, kernel_size=self.avg_pooling_size, stride=1).view(features.size(0), -1)
        out = self.classifier(out)

        out_seg = self.seg_classifier(features)
        out_seg = F.relu(out_seg, inplace=True)
        out_seg = F.tanh(out_seg)

        return out, out_seg


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_seg) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        target_seg = target_seg.cuda(non_blocking=True)

        # compute output
        output, output_seg = model(input)

        # compute loss
        loss1 = criterion(output, target)
        loss2 = criterionMSE(output_seg, target_seg)

        # compose loss
        if args.do_classify and args.do_seg:
            loss = loss1 + loss2
        elif args.do_classify:
            loss = loss1
        elif args.do_seg:
            loss = loss2

        if args.acc_classify:
            prec1 = accuracy(output, target, topk=(1,))
            prec1 = prec1[0].cpu().data.numpy()[0]
        else:
            output_max = F.max_pool2d(output_seg, (args.avg_pooling_height, args.avg_pooling_width))
            output_max = output_max.cpu().data.numpy()
            target = target.cpu().data.numpy()

            output_max = [1.0 if o > 0.5 else 0.0 for o in output_max]
            prec1 = np.average(np.equal(output_max, target).astype(np.float)) * 100

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        if args.train_per_loss and args.do_classify and args.do_seg:
            if args.do_classify:
                optimizer.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer.step()

                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
            elif args.do_seg:
                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                optimizer.step()

                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch,
                                                                  i,
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time,
                                                                  loss=losses,
                                                                  top1=top1))


def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        target_index_output = []
        target_index_target = []
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output, output_seg = model(input)

            if args.acc_classify:
                prec1 = accuracy(output, target, topk=(1,))
                prec1 = prec1[0].cpu().data.numpy()[0]

                # for auroc
                output_cpu = output.squeeze().cpu().data.numpy()
                output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu])  # convert to probability
            else:
                output_max = F.max_pool2d(output_seg, (args.avg_pooling_height, args.avg_pooling_width))
                output_max_cpu = output_max.cpu().data.numpy()
                target_cpu = target.cpu().data.numpy()

                output_max_cpu = [1.0 if o > 0.5 else 0.0 for o in output_max_cpu]
                prec1 = np.average(np.equal(output_max_cpu, target_cpu).astype(np.float)) * 100

                # for auroc
                output_cpu = output_max.squeeze().cpu().data.numpy()

            # --------------------------------------
            # for auroc get value from target index
            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target.cpu().data.numpy(), args.target_index).astype(np.int))
            # --------------------------------------

            # measure accuracy and record loss
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      top1=top1))

    auc, roc = compute_auroc(target_index_output, target_index_target)
    save_auroc(auc, roc, os.path.join(args.result, str(epoch) + '.png'))

    print(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))
    print(' * auc@1 {auc:.3f}'.format(auc=auc))

    return top1.avg


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            img = img.resize((args.resize_image_width, args.resize_image_height))
            return img


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    model = densenet_multi(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                           num_classes=args.num_classes, channels=args.channels, avg_pooling_size=avg_pool_size)

    train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # start main loop
    main(args, model, pil_loader, pil_loader, normalize, train_dataset=MultiTaskDataset,
         train_model=train_model, validate_model=validate_model,
         train_transforms=train_transforms, val_transforms=val_transforms, optimizer=optimizer)
