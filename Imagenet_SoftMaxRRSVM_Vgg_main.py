import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import py_utils.dir_utils as dir_utils
# import torchvision.models as models

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
from models.imagenet.res_vgg import vgg16_bn
import RRSVM.RRSVM as RRSVM
import sys

parser = argparse.ArgumentParser(description='PyTorch ImageNet VGG-BN Training')


parser.add_argument('--finetune', '-f', action='store_true', help='use pre-trained model to finetune')
parser.add_argument("--gpu_id", default=None, type=str)
parser.add_argument('--positive_constraint', '-p', action='store_true', help='positivity constraint')
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256 for others, 32 for Inception-V3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--n_epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='verbose mode, if not, saved in log.txt')
parser.add_argument('--baseline', '-b', dest='baseline', action='store_true', help='using baseline')

best_prec1 = 0

# def main():
#     global args, best_prec1
args = parser.parse_args()

if args.baseline:
    identifier = 'VggBn'
else:
    identifier = 'SoftMaxRRSVMVggBn'
if args.finetune:
    print("=> using pre-trained model")
    pretrained = True
else:
    print("=> creating model from new")
    pretrained = False

if args.baseline:
    useRRSVM = False
else:
    useRRSVM = True

model = vgg16_bn(pretrained, useRRSVM=useRRSVM)

print("Number of Params in {:s}\t{:d}".format(identifier, sum([p.data.nelement() for p in model.parameters()])))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,  model.parameters()), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

p_constraint = False
if args.positive_constraint and not args.baseline:
    p_constraint = True

use_cuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGpu)

if use_cuda:
    if args.multiGpu:
        if args.gpu_id is None: # using all the GPUs
            device_count = torch.cuda.device_count()
            print("Using ALL {:d} GPUs".format(device_count))
            model = nn.DataParallel(model, device_ids=[i for i in range(device_count)]).cuda()
        else:
            print("Using GPUs: {:s}".format(args.gpu_id))
            device_ids = [int(x) for x in args.gpu_id]
            model = nn.DataParallel(model, device_ids=device_ids).cuda()


    else:
        torch.cuda.set_device(int(args.gpu_id))
        model.cuda()

    criterion.cuda()
    cudnn.benchmark = True

# global save_dir
save_dir = './snapshots/ImageNet_{:s}'.format(identifier)
if args.positive_constraint:
    save_dir = save_dir + '_p'
if args.finetune:
    save_dir = save_dir + '_finetune'

save_dir = dir_utils.get_dir(save_dir)
if not args.verbose:
        log_file = os.path.join(save_dir, 'log.txt')
        sys.stdout = open(log_file, "w")
        args.print_freq *= 10

# optionally resume from a checkpoint
if args.resume:

    # if os.path.isfile(args.resume):
    ckpt_filename = 'model_best.ckpt.t7'
    assert os.path.isfile(os.path.join(save_dir, ckpt_filename)), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(os.path.join(save_dir, ckpt_filename), map_location=lambda storage, loc: storage)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['prec1']
    model.load_state_dict(checkpoint['state_dict'])
    # TODO: check how to load optimizer correctly
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loading checkpoint '{}', epoch: {:d}".format(ckpt_filename, args.start_epoch))

else:
    print('==> Training with NO History..')
    if os.path.isfile(os.path.join(save_dir, 'log.txt')):
        os.remove(os.path.join(save_dir, 'log.txt'))

user_root = os.path.expanduser('~')
dataset_path = os.path.join(user_root, 'datasets/imagenet12')
traindir = os.path.join(dataset_path, 'train')
valdir = os.path.join(dataset_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


def train(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adjust_learning_rate(optimizer, epoch, args.finetune)

    if p_constraint:
        positive_clipper = RRSVM.RRSVM_PositiveClipper()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output= model(input_var)
        loss = criterion(output, target_var)
        # loss_aux = criterion(output, target_var)
        # TODO: here check how to merge aux loss
        # t_loss = loss + loss_aux

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if p_constraint and positive_clipper.frequency % (i+1) == 0:
            model.apply(positive_clipper)
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        # #DBUGE
        # if i % 100 == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'prec1': prec1,
        #         'prec5': prec5,
        #         'optimizer': optimizer.state_dict(),
        #     }, False, filename=os.path.join(save_dir, '{:04d}_checkpoint.pth.tar'.format(epoch)))


def validate(epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, isFinetune=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if isFinetune:
        every_n = 2
    else:
        every_n = 30

    lr = args.lr * (0.1 ** (epoch // every_n))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if args.evaluate:
    validate(val_loader, model, criterion, use_cuda)
    sys.exit(0)

for epoch in range(args.start_epoch, args.n_epochs):
    # if args.distributed:
    #     train_sampler.set_epoch(epoch)
    # adjust_learning_rate(optimizer, epoch, args.finetune)

    # train for one epoch
    train(epoch)

    # evaluate on validation set
    prec1, prec5 = validate(epoch)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'prec1': prec1,
        'prec5': prec5,
        'optimizer': optimizer.state_dict(),
    }, is_best, filename=os.path.join(save_dir, '{:04d}_checkpoint.pth.tar'.format(epoch)))
