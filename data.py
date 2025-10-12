import torch
from torchvision import transforms, datasets

def get_data_loader(folder_path, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print(f"Retrieving dataset from {folder_path}...")
    dataset = datasets.ImageFolder(f'{folder_path}', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         num_workers=1, shuffle=True)
    print("Done retrieving dataset.")

    return loader
