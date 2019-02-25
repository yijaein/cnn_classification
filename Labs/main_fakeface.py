import argparse
import time

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from Official.densenet import densenet121
from Official.main import main
from Official.resnet import resnet18
from Official.utils import *
from Works.utils import compute_auroc, softmax, save_auroc

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
parser.add_argument('--simple_resnet', action='store_true', default=True, help='select resnet or densenet')
parser.add_argument('--result', default='../result_fakeface4', help='path to result')
parser.add_argument('--resize_image_width', type=int, default=160, help='image width')
parser.add_argument('--resize_image_height', type=int, default=256, help='image height')
parser.add_argument('--image_width', type=int, default=160, help='image crop width')
parser.add_argument('--image_height', type=int, default=256, help='image crop height')
parser.add_argument('--avg_pooling_width', type=int, default=5, help='average pooling width')
parser.add_argument('--avg_pooling_height', type=int, default=8, help='average pooling height')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--target_index', default=1, type=int, help='target index')
args = parser.parse_args()


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            img = img.resize((args.resize_image_width, args.resize_image_height))
            return img


def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
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
            output = model(input)
            loss = criterion(output, target)

            # --------------------------------------
            # for auroc get value from target index
            output_cpu = output.squeeze().cpu().data.numpy()
            target_cpu = target.cpu().data.numpy()
            output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu]) # convert to probability
            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target_cpu, args.target_index).astype(np.int))
            # --------------------------------------

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses,
                                                                      top1=top1))

    auc, roc = compute_auroc(target_index_output, target_index_target)
    save_auroc(auc, roc, os.path.join(args.result, str(epoch) + '.png'))

    print(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))
    print(' * auc@1 {auc:.3f}'.format(auc=auc))

    return top1.avg


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    if args.simple_resnet:
        model = resnet18(num_classes=args.num_classes, channels=args.channels, avg_pooling_size=avg_pool_size)
    else:
        model = densenet121(num_classes=args.num_classes, channels=args.channels, avg_pooling_size=avg_pool_size)

    train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # start main loop
    main(args, model, pil_loader, pil_loader, normalize, validate_model=validate_model,
         train_transforms=train_transforms, val_transforms=val_transforms)
