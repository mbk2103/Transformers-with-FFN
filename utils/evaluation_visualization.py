# utils/evaluation_visualization.py
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class EvaluationVisualization:
    counter = 1
    save_dir = '/content/drive/MyDrive/NetworksFromScratch/Transformer-withFFN-from-Scratch'

    @staticmethod
    def plot_loss_accuracy(train_losses, val_losses, accuracies, epoch):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        filename = os.path.join(EvaluationVisualization.save_dir, f'plot_confusion_matrix_{EvaluationVisualization.counter}_epoch_{epoch}.png')
        plt.savefig(filename)
        plt.show()

        EvaluationVisualization.counter += 1

    @staticmethod
    def plot_confusion_matrix(conf_matrix, classes, epoch):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        filename = os.path.join(EvaluationVisualization.save_dir, f'plot_confusion_matrix_{EvaluationVisualization.counter}_epoch_{epoch}.png')
        plt.savefig(filename)
        plt.show()

        EvaluationVisualization.counter += 1

def calculate_metrics(labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')

        return accuracy, f1, recall, precision
    
def evaluate_and_visualize(model, dataloader, criterion, device, classes, epoch):
        model.eval()
        all_labels = []
        all_predictions = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the input tensor
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        conf_matrix = confusion_matrix(all_labels, all_predictions)
        EvaluationVisualization.plot_confusion_matrix(conf_matrix, classes, epoch=epoch)

        accuracy, f1, recall, precision = calculate_metrics(all_labels, all_predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')

        return avg_loss, accuracy, conf_matrix  # Return the confusion matrix


