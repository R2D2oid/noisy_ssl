import torchvision

cifar10_crop_size = 32
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]

train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(cifar10_crop_size, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=cifar10_mean,
        std=cifar10_std,
    )
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cifar10_crop_size, cifar10_crop_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=cifar10_mean,
        std=cifar10_std,
    )
])