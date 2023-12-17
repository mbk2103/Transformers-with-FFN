# Transformers-with-FFN
# Vision Transformer with Feedforward Network (FFN)

This project implements a Vision Transformer with a Feedforward Network (FFN) for enhanced performance. The combination of Vision Transformers and FFN aims to accelerate the model's performance, particularly in vision-related tasks.

## Project Structure

The project is organized into several files:

- **main.py**: Contains the main training loop and inference code.
- **factories/vision_transformer_factory.py**: Factory class for creating the Vision Transformer model and data loaders.
- **models/ffn_block.py**: Implementation of the Feedforward Network (FFN) block.
- **models/vision_transformer.py**: Implementation of the Vision Transformer model.
- **utils/evaluation_visualization.py**: Utility functions for evaluating and visualizing the model's performance.

## Usage

To train the Vision Transformer model, run the `main()` function in `main.py`. The trained model will be saved to a specified file path.

```bash
python main.py
```

To perform inference on a single image, uncomment and modify the example code in the main() function.

## Model Configuration

- Input Dimension: 28x28
- Number of Classes: 10
- Number of Blocks: 4
- Hidden Dimension: 256
- FFN Hidden Dimension: 512
- Dropout Probability: 0.1
- Batch Size: 128
- Learning Rate: 0.0001
- Number of Epochs: 10

### Data Loading
The project uses the MNIST dataset. Data loading configurations can be found in the VisionTransformerFactory class.

### Trained Model
The trained Vision Transformer model is saved to the file vision_transformer_model.pth.

### Inference
To perform inference on a single image, use the example code in the main() function. Ensure that the model is loaded from the saved state dict.

### Visualization
The project includes visualization tools for monitoring the model's performance. Confusion matrices are saved to the path/to/project directory.

## Dependencies
- PyTorch
- torchvision
- Pillow
- scikit-learn
- seaborn
- matplotlib

Install dependencies using:
```bash
pip install torch torchvision Pillow scikit-learn seaborn matplotlib
```

Feel free to explore and modify the project to suit your needs!


Please note that you might need to adjust the paths, dependencies, and other details based on your specific setup.

