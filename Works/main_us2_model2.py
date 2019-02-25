import argparse
import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from Official.densenet import DenseNet
from Official.densenet import _DenseBlock, _Transition
from Official.main import main
from Official.utils import AverageMeter
from Official.utils import accuracy
from Tools.crop_by_seg import rotate_bound, calculate_angle, find_bounding_square
from Works.data_augmentation import *
from Works.utils import compute_auroc, softmax, save_auroc
from Tools.utils import image_dict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/pngs/KidneyUs_500_png', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--epoch_decay', default=10, type=int, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--result', default='../result_model500', help='path to result')
parser.add_argument('--aspect_ratio', default=False, action='store_true', help='keep image aspect ratio')
parser.add_argument('--resize_image_width', default=224, type=int, help='image width')
parser.add_argument('--resize_image_height', default=224, type=int, help='image height')
parser.add_argument('--image_width', default=224, type=int, help='image crop width')
parser.add_argument('--image_height', default=224, type=int, help='image crop height')
parser.add_argument('--avg_pooling_width', default=7, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=7, type=int, help='average pooling height')
parser.add_argument('--channels', default=1, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--target_index', default=1, type=int, help='target index')
parser.add_argument('--seg_data', default='/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegKidney_v3', help='path for Seg Kidney dataset')
parser.add_argument('--classification_result', default='', help='path for segmentation result')
parser.add_argument('--kidney_crop_padding', default=20, help='')
parser.add_argument('--preprocess_denoise', default=False, action='store_true', help='reduce noise of train/val image')
args = parser.parse_args()

args.data = os.path.expanduser(args.data)
args.seg_data = os.path.expanduser(args.seg_data)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            out = pil_resize(img)

            # denoise
            if args.preprocess_denoise:
                np_out = np.asarray(out)
                np_out = cv2.fastNlMeansDenoising(np_out, None, 10, 7, 21)
                out = Image.fromarray(np.uint8(np_out))

            return out


def pil_resize(img):
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


def log(*message, end='\n'):
    msg = ""
    for m in message:
        msg += str(m) + " "

    print(msg)
    file = os.path.join(args.result, 'detail_log.txt')
    with open(file, 'at') as f:
        f.write(msg + end)


def image_crop_kidney_loader(input_path, seg_path):
    # input_path : non-kidney image
    # seg_path : any seg image
    # output : crop non-kidney image

    padding_size = args.kidney_crop_padding

    # read image in grayscale
    input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    seg_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    seg_image = cv2.resize(seg_image, (input_image.shape[1], input_image.shape[0]))
    # rotate image
    angle = calculate_angle(seg_image)
    seg_image = rotate_bound(seg_image, angle)
    input_image = rotate_bound(input_image, angle)

    # Add black padding to input image and seg image
    seg_image = cv2.copyMakeBorder(seg_image, padding_size, padding_size, padding_size, padding_size, 0)
    input_image = cv2.copyMakeBorder(input_image, padding_size, padding_size, padding_size, padding_size, 0)

    # get white pixel bounding box
    x, y, w, h = find_bounding_square(seg_image, padding=padding_size)

    # crop kidney with padding
    result_image = input_image[int(y):int(y + h), int(x):int(x + w)]

    # cv2 img into pil img
    img_for_pil = Image.fromarray(result_image)

    resized_img = pil_resize(img_for_pil)

    # #debug
    # debug_path = os.path.join(args.result, 'debug')
    # if not os.path.exists(debug_path):
    #     os.makedirs(debug_path)
    # debug_file = os.path.join(debug_path, os.path.split(input_path)[1])
    # cv2.imwrite(debug_file, result_image)

    return resized_img


def getImagesFiles(img_path):
    image_files = list()
    for (path, dir, files) in os.walk(img_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext != '.png' and ext != '.jpg':
                continue
            image_files.append(os.path.join(path, file))
    return image_files


class TrainDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.seg_files = getImagesFiles(args.seg_data)
        self.KIDNEY = 1
        self.NON_KIDNEY = 0

        # gathering crop_kidney and crop_non-kidney(fake) files
        # (sample_path, target, need crop)
        self.kidney_samples = [[item, self.KIDNEY, False] for item in self.getKidneyFiles(root)]
        self.non_kidney_samples = [[item, self.NON_KIDNEY, True] for item in self.getNonKidneyFiles(root)]
        #신장이 있는데 마스크를 가지고 있는것, 신장과 유사한 사진
        log('Train count samples', len(self.kidney_samples), len(self.non_kidney_samples))

        # balance ratio
        self.samples = self.balanceList([self.kidney_samples, self.non_kidney_samples])

    def balanceList(self, samples_list, ratio=[1.0, 1.0]):
        major_samples = samples_list[0]
        minor_samples = samples_list[1:]

        # get count
        cnt_major_item = len(major_samples)

        # slice sample
        if ratio is not None:
            cnt_minor_item = [int(r * cnt_major_item) for r in ratio[1:]]
            for idx, samples in enumerate(minor_samples):
                n_slice = min(cnt_minor_item[idx], len(samples))
                random.shuffle(samples)
                minor_samples[idx] = samples[:n_slice]

        # concatenate sample
        balance_samples = list()
        balance_samples += major_samples
        msg = 'Train balance sample ' + str(len(balance_samples))
        for sam in minor_samples:
            random.shuffle(sam)
            balance_samples += sam
            msg += ' ' + str(len(sam))

        # log(msg)

        return balance_samples

    def getKidneyFiles(self, path):
        us_files = image_dict(path)
        seg_files = image_dict(args.seg_data)

        kidney_files = [us_files[file] for file in us_files if file in seg_files]

        return kidney_files

    def getNonKidneyFiles(self, path):
        us_files = image_dict(path)
        seg_files = image_dict(args.seg_data)

        non_kidney_files = [us_files[file] for file in us_files if file not in seg_files]

        return non_kidney_files


    def __getitem__(self, index):
        # shuffle no_seg_image rarely
        if random.randint(0, len(self.non_kidney_samples)) == 0:
            self.samples = self.balanceList(
                [self.kidney_samples, self.non_kidney_samples])

        sample_path, target, need_crop = self.samples[index]

        # for debug
        # sample_path, target, need_crop = self.kidney_samples[0]
        # sample_path, target, need_crop = self.non_kidney_samples[0]

        if need_crop:
            random_seg_path = random.choice(self.seg_files)
            sample = image_crop_kidney_loader(sample_path, random_seg_path)

        else:
            # sample = self.loader(sample_path)
            random_seg_path = random.choice(self.seg_files)
            sample = image_crop_kidney_loader(sample_path, random_seg_path)

        #for debug
        # import matplotlib.pyplot as plt
        # sample = np.array(sample)
        #
        # plt.imshow(sample, cmap='gray')
        # plt.show()
        # exit()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_path

    def __len__(self):
        return len(self.samples)


class ValDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.seg_files = getImagesFiles(args.seg_data)
        self.KIDNEY = 1
        self.NON_KIDNEY = 0

        # gathering crop_kidney and crop_non-kidney(fake) files
        # (sample_path, target, need crop)
        self.kidney_samples = [[item, self.KIDNEY, False] for item in self.getKidneyFiles(root)]
        self.non_kidney_samples = [[item, self.NON_KIDNEY, True] for item in self.getNonKidneyFiles(root)]
        log('Val count samples', len(self.kidney_samples), len(self.non_kidney_samples))

        #Validation balanceList
        self.samples = self.balanceList([self.kidney_samples, self.non_kidney_samples])
        if args.evaluate:
            self.samples = self.kidney_samples
        else:
            self.samples = self.kidney_samples + self.non_kidney_samples

    #validation samples balance
    def balanceList(self, samples_list, ratio=[1.0, 1.0]):
        major_samples = samples_list[0]
        minor_samples = samples_list[1:]

        # get count
        cnt_major_item = len(major_samples)

        # slice sample
        if ratio is not None:
            cnt_minor_item = [int(r * cnt_major_item) for r in ratio[1:]]
            for idx, samples in enumerate(minor_samples):
                n_slice = min(cnt_minor_item[idx], len(samples))
                random.shuffle(samples)
                minor_samples[idx] = samples[:n_slice]

        # concatenate sample
        balance_samples = list()
        balance_samples += major_samples
        msg = 'Train balance sample ' + str(len(balance_samples))
        for sam in minor_samples:
            random.shuffle(sam)
            balance_samples += sam
            msg += ' ' + str(len(sam))

        # log(msg)

        return balance_samples

    def getKidneyFiles(self, path):
        us_files = image_dict(path)
        seg_files = image_dict(args.seg_data)

        kidney_files = [us_files[file] for file in us_files if file in seg_files]

        return kidney_files

    def getNonKidneyFiles(self, path):
        us_files = image_dict(path)
        seg_files = image_dict(args.seg_data)

        non_kidney_files = [us_files[file] for file in us_files if file not in seg_files]

        return non_kidney_files



    def __getitem__(self, index):
        sample_path, target, need_crop = self.samples[index]
        if need_crop:
            random_seg_path = random.choice(self.seg_files)
            sample = image_crop_kidney_loader(sample_path, random_seg_path)

        else:
            random_seg_path = random.choice(self.seg_files)
            sample = image_crop_kidney_loader(sample_path, random_seg_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_path

    def __len__(self):
        return len(self.samples)


def pred(output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        return pred


class densenet_seg(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
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
        out = F.tanh(out)

        return out


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, input_path) in enumerate(train_loader):
        # measure data loading timepil_resize
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

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
            message = ('Epoch: [{0}][{1}/{2}]\t' +
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' +
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})').format(epoch,
                                                                        i,
                                                                        len(train_loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time,
                                                                        loss=losses,
                                                                        top1=top1)
            log(message)


def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cnt_cnt_label = [0] * args.num_classes
    cnt_exact_pred = [0] * args.num_classes

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        target_index_output, target_index_target = list(), list()
        for i, (input, target, input_path) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # --------------------------------------
            # for auroc get value from target index
            output_cpu = output.squeeze().cpu().data.numpy()
            output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu])
            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target.cpu().data.numpy(), args.target_index).astype(np.int))
            # --------------------------------------

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # put together for acc per label
            pred_list = pred(output).cpu().numpy().squeeze()
            target_list = target.cpu().numpy().squeeze()
            for (p, t) in zip(pred_list, target_list):
                cnt_cnt_label[t] += 1
                if p == t:
                    cnt_exact_pred[t] += 1

                pred_list = pred(output).cpu().numpy().squeeze()
                for pred_idx, pred_item in enumerate(pred_list):
                    dst = os.path.join(args.classification_result, 'kidney' if pred_item == 1 else 'no_kidney')

                    if not os.path.exists(dst):
                        os.makedirs(dst)

                    seg_img = input_path[pred_idx]
                    shutil.copy(seg_img, dst)

            if i % print_freq == 0:
                log(('Test: [{0}/{1}]\t' +
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})').format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses,
                                                                      top1=top1))

        auc, roc = compute_auroc(target_index_output, target_index_target)
        # save_auroc(auc, roc, os.path.join(args.result, str(epoch) + '.png'))

        log(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))
        log(' * auc@1 {auc:.3f}'.format(auc=auc))

        for (i, (n_label, n_exact)) in enumerate(zip(cnt_cnt_label, cnt_exact_pred)):
            acc_label = (n_exact / n_label * 100) if n_label > 0 else 0
            log('acc of label {:0d}: {:0.3f}%'.format(i, acc_label))

    return auc


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    model = DenseNet(num_init_features=32, growth_rate=16, block_config=(6, 12, 24, 16), num_classes=args.num_classes,
                     channels=args.channels, avg_pooling_size=avg_pool_size)

    train_transforms = transforms.Compose([transforms.RandomCrop((args.image_height, args.image_width)),
                                           transforms.ToTensor(),
                                           normalize,
                                           ])

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # start main loop
    main(args, model, train_image_loader, valid_image_loader, normalize, optimizer,
         train_dataset=TrainDataset, valid_dataset=ValDataset,
         train_model=train_model, validate_model=validate_model,
         train_transforms=train_transforms)
