import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def ifUseCuda(gpu_id, multiGPU):
    return torch.cuda.is_available() and (gpu_id is not None or multiGPU)


def convertModel2Cuda(model, args):
    use_cuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGPU)
    if use_cuda:
        if args.multiGpu:
            if args.gpu_id is None:  # using all the GPUs
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

        cudnn.benchmark = True

    return  model