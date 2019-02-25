import argparse
import os
import random
import time
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from Official.densenet import _DenseBlock, _Transition
from Official.main import main
from Official.utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='~/data/fakeface4', help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', type=int, default=30, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--simple_resnet', action='store_true', default=False, help='select resnet or densenet')
parser.add_argument('--result', default='../result_fakeface_multi', help='path to result')
parser.add_argument('--resize_image_width', type=int, default=160, help='image width')
parser.add_argument('--resize_image_height', type=int, default=256, help='image height')
parser.add_argument('--image_width', type=int, default=160, help='image crop width')
parser.add_argument('--image_height', type=int, default=256, help='image crop height')
parser.add_argument('--avg_pooling_width', type=int, default=5, help='average pooling width')
parser.add_argument('--avg_pooling_height', type=int, default=8, help='average pooling height')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--multi_data1', metavar='DIR', default='~/data/vggface2_hm400/train', help='dataset path for multi-task learning')
parser.add_argument('--multi_data_num_classes1', default=8362, type=int, help='number of classes')
args = parser.parse_args()

args.multi_data1 = os.path.expanduser(args.multi_data1)


class MultiTaskDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        # read folder and make dataset
        self.samples = []  # (input, target)
        list_dir = os.listdir(self.root)
        list_dir.sort()
        for label, sub_dir in enumerate(list_dir):
            sub_dir = os.path.join(self.root, sub_dir)

            for (path, dir, files) in os.walk(sub_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == '.png' or ext == '.jpg':
                        image_path = os.path.join(path, filename)
                        self.samples.append((image_path, label))

        self.multi_samples1 = []
        list_dir = os.listdir(args.multi_data1)
        list_dir.sort()
        for label, sub_dir in enumerate(list_dir):
            sub_dir = os.path.join(args.multi_data1, sub_dir)

            for (path, dir, files) in os.walk(sub_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == '.png' or ext == '.jpg':
                        image_path = os.path.join(path, filename)
                        self.multi_samples1.append((image_path, label))


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # for multi-task
        mul_path1, mul_target1 = random.choice(self.multi_samples1)
        mul_sample1 = self.loader(mul_path1)
        if self.transform is not None:
            mul_sample1 = self.transform(mul_sample1)
        if self.target_transform is not None:
            mul_target1 = self.target_transform(mul_target1)

        return sample, target, mul_sample1, mul_target1

    def __len__(self):
        return len(self.samples)


class densenet_multi(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, channels=3, num_classes=1000, avg_pooling_size=7):

        super(densenet_multi, self).__init__()

        self.avg_pooling_size = avg_pooling_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
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
        self.mul_classifier1 = nn.Linear(num_features, args.multi_data_num_classes1)  # for multi task

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
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avg_pooling_size, stride=1).view(features.size(0), -1)
        logit = self.classifier(out)
        mul_out1 = self.mul_classifier1(out)  # for multi task
        if self.training:
            return logit, mul_out1
        else:
            return logit


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, input2, target2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        target2 = target2.cuda(non_blocking=True)  # for multi task

        # compute output
        output, output2 = model(input)  # for multi task
        loss = criterion(output, target)
        loss2 = criterion(output2, target2)  # for multi task

        loss = loss + loss2  # for multi task

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

        # compute gradient and do SGD step
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
    if args.simple_resnet:
        raise NotImplementedError()
    else:
        model = densenet_multi(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                               num_classes=args.num_classes, channels=args.channels, avg_pooling_size=avg_pool_size)

    train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # start main loop
    main(args, model, pil_loader, pil_loader, normalize, train_dataset=MultiTaskDataset,
         train_model=train_model, train_transforms=train_transforms, val_transforms=val_transforms)
