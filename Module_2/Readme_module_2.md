# PyTorch Learning - Module 2: Building Your First Neural Network

> From data to deployment: A complete end-to-end workflow for building, training, and saving PyTorch models.

## 🎯 What You'll Learn

This module takes you through the entire lifecycle of a neural network project in PyTorch. You'll go from raw data to a trained, saved model ready for deployment - understanding every step along the way.

## 📖 Module Overview

### The Journey
```
Raw Data → Model Architecture → Training Loop → Evaluation → Model Persistence → Deployment
```

This notebook teaches you how to think like a deep learning engineer by building a complete workflow from scratch.

## 🧠 Core Concepts

### 1. **Understanding the Linear Model**
- **What**: Learn how neural networks start with simple linear transformations
- **Formula**: `y = wx + b` (weight × input + bias)
- **Why it matters**: Foundation for understanding more complex architectures

### 2. **Data Preparation & Splitting**
- Creating synthetic data for experimentation
- **Train-Test Split**: Why we need separate data for training and validation
- Using `torch.arange()` and tensor operations for data generation
- Shape manipulation with `.unsqueeze()` for proper dimensions

### 3. **Building Your First PyTorch Model**
- **Inheriting from `nn.Module`**: The PyTorch way of defining models
- Understanding the model architecture pattern:
  ```python
  class MyModel(nn.Module):
      def __init__(self):      # Define layers
      def forward(self, x):    # Define forward pass
  ```
- **Parameters**: What they are and why they're learnable
- Using `torch.manual_seed()` for reproducibility

### 4. **Making Predictions**
- Forward pass: How data flows through your model
- Understanding model output before training
- The importance of initial random weights

### 5. **The Training Loop - Where Learning Happens**

#### Key Components:
- **Loss Function**: Measuring how wrong your model is
  - `nn.L1Loss()` - Mean Absolute Error
  - `nn.MSELoss()` - Mean Squared Error
- **Optimizer**: The algorithm that updates weights
  - `torch.optim.SGD()` - Stochastic Gradient Descent
  - Learning rate and its impact

#### The Training Workflow:
```python
for epoch in range(epochs):
    # 1. Forward pass - make predictions
    # 2. Calculate loss - how wrong are we?
    # 3. Zero gradients - clear old gradients
    # 4. Backpropagation - calculate new gradients
    # 5. Optimizer step - update weights
```

### 6. **Evaluation & Testing**
- **Train vs Eval Mode**: Understanding `model.train()` and `model.eval()`
- Testing on unseen data
- Measuring model performance
- Why we don't update weights during testing

### 7. **Tracking Progress**
- Logging losses during training
- Creating loss curves with matplotlib
- Visualizing learning progress
- Understanding convergence

### 8. **Model Persistence (Saving & Loading)**

#### Saving Models:
- Creating model directories
- Using `torch.save()` and `state_dict()`
- Best practices for model checkpointing
- **Why state_dict?** Saves learned parameters, not the entire model structure

#### Loading Models:
- Recreating model architecture
- Loading saved weights with `load_state_dict()`
- Continuing training from checkpoints
- Deploying saved models

### 9. **Device-Agnostic Code (CPU vs GPU)**

#### Writing Portable Code:
- Checking device availability
- Using `torch.device()` for flexibility
- Moving models and data to GPU with `.to(device)`
- **Why it matters**: Same code works on any hardware

#### Pattern:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
data.to(device)
```

## 🔄 The Complete Workflow

```mermaid
graph LR
    A[Data Preparation] --> B[Build Model]
    B --> C[Define Loss & Optimizer]
    C --> D[Training Loop]
    D --> E[Evaluation]
    E --> F[Save Model]
    F --> G[Load & Deploy]
```

## 💻 Key Code Patterns

### Creating a Custom Model
```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)
```

### Training Loop Template
```python
model.train()
for epoch in range(epochs):
    y_pred = model(X_train)              # Forward pass
    loss = loss_fn(y_pred, y_train)      # Calculate loss
    optimizer.zero_grad()                 # Reset gradients
    loss.backward()                       # Backpropagation
    optimizer.step()                      # Update weights
```

### Evaluation Loop Template
```python
model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
    loss = loss_fn(y_pred, y_test)
```

## 🛠️ Prerequisites

```python
torch>=2.0.0
matplotlib
numpy
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. **Install dependencies**
```bash
pip install torch matplotlib numpy
```

3. **Run the notebook**
```bash
jupyter notebook Pytorch_Module2.ipynb
```

## 🎓 Learning Outcomes

By the end of this module, you will:

✅ Understand the complete PyTorch workflow from data to deployment  
✅ Build custom neural network models using `nn.Module`  
✅ Implement training and evaluation loops from scratch  
✅ Choose appropriate loss functions and optimizers  
✅ Visualize training progress and debug models  
✅ Save and load trained models  
✅ Write device-agnostic code that runs on CPU or GPU  

## 🧩 Key Insights

> **Training is iteration**: Models learn by repeatedly adjusting weights to minimize error

> **State dict is lightweight**: Only save what's learned, not the entire architecture

> **Device agnostic = flexibility**: Write once, run anywhere (CPU/GPU/TPU)

> **Evaluation mode matters**: Disable dropout and batch norm during inference

## 📊 What's Next?

After mastering this module, you're ready for:
- Working with real datasets (images, text, tabular data)
- Building deeper neural networks (CNNs, RNNs, Transformers)
- Advanced optimization techniques (Adam, learning rate scheduling)
- Data loaders and batching
- More complex loss functions and metrics

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request!

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Understanding Backpropagation](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

---

**Made with 🔥 and PyTorch** | Happy Learning! 🚀
