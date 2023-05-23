import torch
import torchvision

from lightly.data import LightlyDataset, collate

from utils.helper_functions import yaml_loader, generate_embeddings, plot_knn_examples
from tools.simple_clr import SimCLRModel


general_dict = yaml_loader('configs/general.yaml')

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((general_dict['input_size'], general_dict['input_size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize["mean"],
            std=collate.imagenet_normalize["std"],
        ),
    ]
)

dataset_test = LightlyDataset(input_dir=general_dict['path_to_test_data'], transform=test_transforms)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=general_dict['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=general_dict['num_workers'],
)

if __name__ == '__main__':
    model = SimCLRModel.load_from_checkpoint("lightning_logs/version_6/checkpoints/epoch=9-step=100.ckpt")
    model.eval()
    embeddings, filenames = generate_embeddings(model, dataloader_test)
    plot_knn_examples(embeddings, filenames)