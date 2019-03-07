import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from Official.densenet import _DenseBlock, _Transition
from Official.main import main
from Official.utils import AverageMeter, save_tensor_image
from Works.data_augmentation import *
from Works.utils import compute_psnr, save_log_graph, post_processing
from unet import UNet

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_IDS = [1]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--epoch_decay', default=80, type=int, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--result', default='../result_Densenet264', help='path to result')
parser.add_argument('--aspect_ratio', default=True, action='store_true', help='keep image aspect ratio')
parser.add_argument('--resize_image_width', default=512, type=int, help='image width')
parser.add_argument('--resize_image_height', default=512, type=int, help='image height')
parser.add_argument('--image_width', default=512, type=int, help='image crop width')
parser.add_argument('--image_height', default=512, type=int, help='image crop height')
parser.add_argument('--avg_pooling_width', default=32, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=32, type=int, help='average pooling height')
parser.add_argument('--channels', default=1, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--seg_data', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3', help='path for segmentation dataset')
parser.add_argument('--seg_result', default='/home/bong6/lib/robin_yonsei3/result_us1_model1/Densenet264', help='path for segmentation result')
parser.add_argument('--post_processing', default=False, action='store_true', help='do post-processing seg result')
parser.add_argument('--seg_only_kidney', default=False, action='store_true', help='segment only kidney dataset')
parser.add_argument('--unet', default=False, action='store_true',help="enable unet")

args = parser.parse_args()

args.seg_data = os.path.expanduser(args.seg_data)

# Think better way to handle this MSE
criterionMSE = nn.MSELoss().cuda()
criterionCros = nn.CrossEntropyLoss().cuda()


# Think better way to handel seg_pil_loader
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')

            # set resize_image_size
            resize_image_size = (args.resize_image_width, args.resize_image_height)
            if args.aspect_ratio:
                img.thumbnail(resize_image_size)
                offset = ((resize_image_size[0] - img.size[0]) // 2, (resize_image_size[1] - img.size[1]) // 2)
                back = Image.new("RGB" if args.channels == 3 else 'L', resize_image_size, "black")
                back.paste(img, offset)
                out = back
            else:
                out = img.resize(resize_image_size)

            return out


def train_image_loader(path):
    out = pil_loader(path)

    out = np.array(out)
    out = random_gamma(out, min=0.5, max=1.5)
    # 히스토그램 균일화
    out = random_clahe(out, min=0, max=0.75)
    out = random_sharpen(out, max_ratio=0.5)
    out = Image.fromarray(np.uint8(out))

    return out


def valid_image_loader(path):
    out = pil_loader(path)

    out = np.array(out)
    out = apply_clahe(out, 0.5)
    out = Image.fromarray(np.uint8(out))

    return out


def seg_pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L')

            # set resize_image_size
            resize_image_size = (args.avg_pooling_width, args.avg_pooling_height)
            if args.aspect_ratio:
                img.thumbnail(resize_image_size)
                offset = ((resize_image_size[0] - img.size[0]) // 2, (resize_image_size[1] - img.size[1]) // 2)
                back = Image.new('L', resize_image_size, "black")
                back.paste(img, offset)
                img = back
            else:
                img = img.resize(resize_image_size)

            return img


class SegDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.seg_loader = seg_pil_loader
        self.transform = transform
        self.target_transform = target_transform
        self.seg_transform = transforms.Compose([transforms.ToTensor()])

        black_image_path = 'black.png'
        resize_image_size = (args.avg_pooling_width, args.avg_pooling_height)
        black = Image.new('L', resize_image_size, "black")
        black.save(black_image_path)

        # read folder and make dataset
        self.samples = []  # (input, target, seg_image_path)
        self.seg_image = []
        self.no_seg_image = []

        seg_image_path = dict()
        for (path, dir, files) in os.walk(args.seg_data):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.png' or ext == '.jpg':
                    seg_image_path[filename] = os.path.join(path, filename)

        for (path, dir, files) in os.walk(self.root):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.png' or ext == '.jpg':
                    image_path = os.path.join(path, filename)
                    if filename in seg_image_path:
                        self.seg_image.append((image_path, 0, seg_image_path[filename]))
                    else:
                        self.no_seg_image.append((image_path, 1, black_image_path))
        # set samples

        if args.seg_only_kidney:
            self.samples = self.seg_image
            print(len(self.samples))
        else:
            self.samples = self.seg_image + self.no_seg_image
            print(len(self.samples))

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

        return sample, target, seg_sample, path

    def __len__(self):
        return len(self.samples)


class TrainSegDataset(SegDataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        super(TrainSegDataset, self).__init__(root, loader, transform, target_transform)

        # balance data distribution
        if not args.seg_only_kidney:
            if len(self.no_seg_image) > len(self.seg_image):
                #신장보다 신장이 아닌 이미지가 있어서 신장이 아닌 이미지를 골고루 학습시키기 위해서
                random.shuffle(self.no_seg_image)
                self.samples = self.seg_image + self.no_seg_image[:len(self.seg_image)]
            else:
                self.samples = self.seg_image + self.no_seg_image

    def __getitem__(self, index):
        # shuffle no_seg_image rarely
        if random.randint(0, len(self.no_seg_image)) == 0 and not args.seg_only_kidney:
            if len(self.no_seg_image) > len(self.seg_image):
                random.shuffle(self.no_seg_image)
                self.samples = self.seg_image + self.no_seg_image[:len(self.seg_image)]
            else:
                self.samples = self.seg_image + self.no_seg_image

        sample, target, seg_sample, path = super(TrainSegDataset, self).__getitem__(index)
        return sample, target, seg_sample, path

class densenet_seg(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 64, 48),
                 num_init_features=64, bn_size=4, drop_rate=0, channels=3, num_classes=1000, avg_pooling_size=7):

        super(densenet_seg, self).__init__()

        self.avg_pooling_size = avg_pooling_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
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
        '''기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다. 
        이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다. 
        따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.
        Batch Normalization의 경우 자체적인 regularization 효과가 있다. 이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며, 
        나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.) .
        Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다'''
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # for segmentation
        self.seg_classifier = nn.Conv2d(num_features, 1, kernel_size=1, bias=False)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        out = self.seg_classifier(out)
        out = F.relu(out, inplace=True)
        out = nn.functional.tanh(out)

        return out


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_seg, filename) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        target_seg = target_seg.cuda(non_blocking=True)

        # compute output
        output_seg = model(input)

        # compute loss
        if args.unet:
            loss = criterionCros(output_seg, target_seg)
        else:
            loss = criterionMSE(output_seg, target_seg)

        # compute psnr
        psnr = compute_psnr(loss)
        psnrs.update(psnr)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

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
                  'Psnr {psnr:.4f} ({psnr:.4f})'.format(epoch,
                                                        i,
                                                        len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        loss=losses,
                                                        psnr=psnr))


def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_seg, filename) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            target_seg = target_seg.cuda(non_blocking=True)

            # compute output
            output_seg = model(input)

            # compute loss
            # add crossentropy
            if args.unet:
                loss = criterionCros(output_seg,target_seg)
            else:
                loss = criterionMSE(output_seg, target_seg)

            # measure accuracy and record loss
            psnrs.update(compute_psnr(loss))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # save segmentation result
            if args.seg_result != '':
                name = [os.path.split(f)[1] for f in filename]
                save_tensor_image(output_seg, name, args.seg_result)

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Psnr {psnr.val:.4f} ({psnr.avg:.4f})'.format(i,
                                                                    len(val_loader),
                                                                    batch_time=batch_time,
                                                                    loss=losses,
                                                                    psnr=psnrs))

    print(' * Loss@1 {loss.avg:.4f} Psnr@1 {psnr.avg:.4f} at Epoch {epoch:0}'.format(loss=losses, psnr=psnrs,
                                                                                     epoch=epoch))

    # save log graph
    # It's not a good location, but it's practical.

    # save_log_graph(log=os.path.join(args.result, 'log.txt'))

    # do post process seg_result

    if args.post_processing:
        for filename in os.listdir(args.seg_result):
            filename = os.path.join(args.seg_result, filename)
            with open(filename, 'rb') as f:
                with Image.open(f) as img:
                    # do post processing
                    img = np.array(img)
                    img = post_processing(img)
                    #add resize
                    img = cv2.resize(img, dsize=(512, 512))
                    img = Image.fromarray(img)
                    # save post processed result
                    img.save(filename)

    return psnrs.avg


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)

    if args.unet :
        args.avg_pooling_width = 512
        args.avg_pooling_height= 512

        #add U-net model
        model = UNet(n_channels=args.channels, n_classes=1)

    else:
        model = densenet_seg(num_init_features=32, growth_rate=16, block_config=(6, 12, 64, 48),
                         num_classes=args.num_classes, channels=args.channels, avg_pooling_size=avg_pool_size)

    train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # start main loop
    main(args, model, train_image_loader, valid_image_loader, normalize, optimizer,
         train_dataset=TrainSegDataset, valid_dataset=SegDataset,
         train_model=train_model, validate_model=validate_model,
         train_transforms=train_transforms, val_transforms=val_transforms)
