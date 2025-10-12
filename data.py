import torch
from torchvision import transforms, datasets

def get_data_loader(folder_name, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_folder_location = ('/content/gdrive/MyDrive/Colab Notebooks'
                            '/APS360/Lab 3/Data')

    print("Retrieving training dataset...")
    train_dataset = datasets.ImageFolder(
        f'{data_folder_location}/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    print("Got training dataset.")

    print("Retrieving validation dataset...")
    val_dataset = datasets.ImageFolder(
        f'{data_folder_location}/val', transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    print("Got validation dataset.")

    print("Retrieving testing dataset...")
    test_dataset = datasets.ImageFolder(
        f'{data_folder_location}/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    print("Got testing dataset.")

    return train_loader, val_loader, test_loader
