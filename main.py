import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from utils.evaluation_visualization import EvaluationVisualization, evaluate_and_visualize
from factories.vision_transformer_factory import VisionTransformerFactory
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels, = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, accuracy, conf_matrix = evaluate_and_visualize(model, val_loader, criterion, device, classes, epoch)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        EvaluationVisualization.plot_loss_accuracy(train_losses, val_losses, accuracies, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Hyperparameters
  input_dim = 28*28
  num_classes = 10
  num_blocks = 4
  hidden_dim = 256
  ffn_hidden_dim = 512
  dropout_prob = 0.1
  batch_size = 128
  learning_rate = 0.0001
  num_epochs = 10

  global classes

  classes = [str(i) for i in range(10)] # MNIST Class Labels

  # Create model
  model = VisionTransformerFactory.create_model(
      input_dim, num_classes, num_blocks, hidden_dim, ffn_hidden_dim, dropout_prob
  )
  model.to(device)

  # Create data loaders
  train_loader, val_loader = VisionTransformerFactory.create_dataloader(batch_size)

  # Loss and Optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

  # Train Loop
  train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

  # Save the trained model
  torch.save(model.state_dict(), '/content/drive/MyDrive/NetworksFromScratch/Transformer-withFFN-from-Scratch/vision_transformer_model.pth')

  # Load the trained model for inference
  model.load_state_dict(torch.load('/content/drive/MyDrive/NetworksFromScratch/Transformer-withFFN-from-Scratch/vision_transformer_model.pth'))
  model.eval()

  # Example: Inference on a single image
#   example_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#   example_image = example_transform(Image.open('/content/drive/MyDrive/NetworksFromScratch/Inputs/mnist5.jpg')).unsqueeze(0).to(device)
#   with torch.no_grad():
#       output = model(example_image.view(example_image.size(0), -1))  # Flatten the input tensor for inference
#   predicted_class = torch.argmax(output, dim=1).item()
#   print(f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
  main()