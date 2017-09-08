import torchvision.transforms as o_transforms

transform_train = o_transforms.Compose([
    o_transforms.RandomCrop(32, padding=4),
    o_transforms.RandomHorizontalFlip(),
    o_transforms.ToTensor(),
    o_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = o_transforms.Compose([
    o_transforms.ToTensor(),
    o_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
