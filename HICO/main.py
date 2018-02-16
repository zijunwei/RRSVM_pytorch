import argparse
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/RRSVM_pytorch')
sys.path.append(project_root)
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import py_utils.dir_utils as dir_utils
import Network
from PtUtils import cuda_model
import RRSVM.RRSVM as RRSVM
import HICODataLoader2 as HICODataLoader
import Metrics
import progressbar
import numpy as np
import pickle as pkl


parser = argparse.ArgumentParser(description='PyTorch HICO Training With ResNet101')
parser.add_argument("--gpu_id", default='01', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256 for others, 32 for Inception-V3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

best_mAP = 0


def main():
    global args, best_mAP
    args = parser.parse_args()
    useRRSVM = True
    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    model = Network.getRes101Model(eval=False, gpu_id=args.gpu_id, multiGpu=args.multiGpu, useRRSVM=useRRSVM)

    print("Number of Params in ResNet101\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    #TODO: add weight
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor([10,1]).cuda())
    # you need to modify this to satisfy the papers w_p =10 and w_n = 1
    criterion = Network.WeightedBCEWithLogitsLoss(weight=torch.FloatTensor([1, 10]))
    # criterion = nn.BCEWithLogitsLoss()
    if use_cuda:
        criterion.cuda()
        cudnn.benchmark = True



    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,  model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.(filter(lambda p:p.requires_grad,  model.parameters()), args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[20, 40])


    global save_dir
    save_dir = './snapshots/HICO_ResNet101_wBCE'
    save_dir = dir_utils.get_dir(save_dir)

    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        ckpt_filename = 'model_best.pth.tar'
        assert os.path.isfile(os.path.join(save_dir, ckpt_filename)), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(os.path.join(save_dir, ckpt_filename), map_location=lambda storage, loc: storage)
        # args.start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        args.start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # TODO: check how to load optimizer correctly
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loading checkpoint '{}', epoch: {:d}, current Precision: {:.04f}".format(ckpt_filename, args.start_epoch, best_mAP))


    train_loader = torch.utils.data.DataLoader(HICODataLoader.HICODataset(split='train', transform=HICODataLoader.HICO_train_transform()),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(HICODataLoader.HICODataset(split='test', transform=HICODataLoader.HICO_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print("Evaluation Only")
        mAP, loss = validate(test_loader, model, criterion, use_cuda)

        return
    avg_train_losses = []
    avg_test_losses = []
    for epoch in range(args.start_epoch, args.epochs):

        lr_scheduler.step(epoch)
        print('Epoch\t{:d}\t LR lr {:.5f}'.format(epoch,optimizer.param_groups[0]['lr']))

        # train for one epoch
        _, avg_train_loss = train(train_loader, model, criterion, optimizer, epoch, use_cuda)

        # evaluate on validation set
        mAP, avg_test_loss = validate(test_loader, model, criterion, use_cuda)

        # remember best prec@1 and save checkpoint
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'mAP': mAP,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(save_dir, '{:04d}_checkpoint.pth.tar'.format(epoch)))
        avg_train_losses.append(avg_train_loss)
        avg_test_losses.append(avg_test_loss)

        loss_record = {'train': avg_train_losses, 'test': avg_test_losses}
        with open(os.path.join(save_dir, 'loss.pkl'), 'wb') as handle:
            pkl.dump(loss_record, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # with open('filename.pickle', 'rb') as handle:
        #     b = pickle.load(handle)
        #
        # print a == b



def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5 = AverageMeter()

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
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # s_mAP = Metrics.calculate_mAP(output.data.cpu().numpy(), target_var.data.cpu().numpy())
        # s_mAP = Metrics.meanAP(output.data.cpu().numpy(), target_var.data.cpu().numpy())

        # prec1, prec5 = Metrics.accuracy(output.data, target, topk=(1, 5))
        prec = Metrics.match_accuracy(output.data, target)
        top5.update(prec, input.size(0))
        losses.update(loss.data[0], input.size(0))

        # mAP.update(s_mAP, input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLR {lr:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {mAP.val:.3f} ({mAP.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=optimizer.param_groups[0]['lr'], loss=losses, mAP=top5))

    return top5.avg, losses.avg


def validate(val_loader, model, criterion, useCuda):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # mAP = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = progressbar.ProgressBar(max_value=len(val_loader))
    y_pred = []
    y_true = []
    for i, (input, target) in enumerate(val_loader):
        pbar.update(i)
        if useCuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # sigmoid_output = F.softmax(output, dim=1)
        sigmoid_output = F.sigmoid(output)

        y_pred.append(sigmoid_output.data.cpu().numpy())
        y_true.append(target_var.data.cpu().numpy())

        # s_mAP = Metrics.meanAP(output.data.cpu().numpy().transpose(), target_var.data.cpu().numpy().transpose())
        # s_mAP = Metrics.meanAP(output.data.cpu().numpy(), target_var.data.cpu().numpy())

        losses.update(loss.data[0], input.size(0))
        # mAP.update(s_mAP, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    mAP = Metrics.calculate_mAP(y_pred, y_true)
    # mAP, _ = Metrics.mAPNips2017(y_pred, y_true)

    print(' * mAP  {:.3f}\t *loss {loss.avg:.4f}'
          .format(mAP, loss=losses))

    return mAP, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_directory = os.path.dirname(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_directory, 'model_best.pth.tar'))


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

if __name__ == '__main__':
    main()