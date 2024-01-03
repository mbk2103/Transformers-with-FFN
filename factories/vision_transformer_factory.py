from models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class VisionTransformerFactory:
  @staticmethod
  def create_model(input_dim, num_classes, num_blocks, hidden_dim, ffn_hidden_dim, dropout_prob=0.1):
    return VisionTransformer(input_dim, num_classes, num_blocks, hidden_dim, ffn_hidden_dim, dropout_prob)

  @staticmethod
  def create_dataloader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    dataset = datasets.MNIST(root="path/to/here", train=True, transform = transform, download = True)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
