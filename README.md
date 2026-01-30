# PyTorch Learning - Module 1

A comprehensive introduction to PyTorch fundamentals covering tensors, operations, and core concepts for deep learning.

## 📚 Overview

This notebook provides a hands-on introduction to PyTorch, focusing on tensor operations and fundamental concepts needed to build neural networks. Perfect for beginners getting started with PyTorch and deep learning.

## 🎯 Topics Covered

### 1. **Introduction to Tensors**
- Understanding scalars, vectors, matrices, and higher-dimensional tensors
- Creating tensors with `torch.tensor()`
- Tensor dimensions (`.ndim`), shape (`.shape`), and size (`.size()`)
- Indexing and accessing tensor elements

### 2. **Random Tensors**
- Creating random tensors for neural network initialization
- Understanding the role of random tensors in deep learning
- Random tensor generation methods

### 3. **Tensor Creation Methods**
- Creating tensors with specific ranges
- Zeros and ones tensors
- Tensor-like operations

### 4. **Tensor Datatypes**
- Understanding PyTorch data types
- Converting between datatypes
- Common issues: The 3 big tensor errors
  - Wrong datatype
  - Wrong shape
  - Wrong device (CPU vs GPU)

### 5. **Tensor Operations**
- Basic arithmetic operations
- Element-wise multiplication
- Matrix multiplication (`matmul`)
- Broadcasting

### 6. **Tensor Aggregation**
- Finding min, max, mean, sum
- Positional min/max using `argmin()` and `argmax()`
- Statistical operations on tensors

### 7. **Tensor Manipulation**
- **Reshaping**: Changing tensor dimensions with `.reshape()`
- **Stacking**: Combining tensors with `torch.stack()`
- **Squeezing**: Removing single dimensions with `.squeeze()`
- **Unsqueezing**: Adding dimensions with `.unsqueeze()`
- **Permute**: Swapping dimensions with `.permute()`

### 8. **NumPy Integration**
- Converting between PyTorch tensors and NumPy arrays
- `torch.from_numpy()` for NumPy to tensor conversion
- `.numpy()` for tensor to NumPy conversion

### 9. **PyTorch Reproducibility**
- Setting random seeds for reproducible results
- Understanding randomness in neural networks

### 10. **Device-Agnostic Code**
- Running PyTorch on CPU and GPU
- Writing code that works on any device
- Checking device availability
- Moving tensors between devices

## 🛠️ Requirements

```python
torch>=2.0.0
numpy
pandas
matplotlib
```

## 🚀 Getting Started

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install torch numpy pandas matplotlib
```

3. Open the notebook:
```bash
jupyter notebook Pytorch_learning_Module1.ipynb
```

Or use Google Colab to run it directly in your browser.

## 💡 Key Takeaways

- Tensors are the fundamental data structure in PyTorch
- Understanding tensor shapes and dimensions is crucial for building neural networks
- PyTorch tensors can seamlessly move between CPU and GPU
- Random tensors form the initial weights in neural networks
- Tensor operations are optimized for GPU acceleration

## 📝 Notes

This module focuses on the foundational concepts. The notebook includes:
- Clear code examples for each concept
- Practical demonstrations of tensor operations
- Common pitfalls and how to avoid them

## 🎓 Next Steps

After completing this module, you'll be ready to:
- Build simple neural networks
- Understand PyTorch's computational graph
- Work with PyTorch datasets and dataloaders
- Implement training loops

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements or additional examples!

## 📄 License

This project is open source and available for educational purposes.

---

**Happy Learning! 🚀**
