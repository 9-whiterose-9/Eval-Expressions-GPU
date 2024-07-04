# Evaluating Expressions - Parallel Computing with GPU


The objective of this project is to develop a GPU-accelerated version of a sequential algorithm for evaluating mathematical expressions, utilizing CUDA for parallel processing. The goal is to minimize the Mean Square Error (MSE) of predictions using candidate solutions, leveraging the computational power of the GPU to achieve significant speedups.

## Goals

1. **Parallelize the sequential expression evaluation algorithm** using CUDA.
2. **Optimize the performance** by minimizing kernel calls and data transfers.
3. **Evaluate the performance** of the GPU implementation compared to the CPU implementation.

## Key Features

1. **CUDA-based Parallelization**:
    - Utilizes Numba and PyCUDA for GPU acceleration.
    - Dynamically generates CUDA kernel code from mathematical expressions.
    - Evaluates multiple expressions in parallel.

2. **Memory Management**:
    - Efficiently handles data transfer between CPU and GPU.
    - Minimizes the number of memory copies and kernel calls.

3. **Performance Optimization**:
    - Fine-tunes the number of threads and blocks for optimal GPU utilization.
    - Implements strategies to reduce branch conflicts and enhance parallelism.

## Fine-tuning for Performance

- **Depth-based Condition**: Switches to sequential execution at recursion depth >= 15.
- **Asynchronous Data Transfer**: Utilizes asynchronous transfers to reduce overhead.

## Tools and Technologies

- **Python**: Programming language used for implementation.
- **Numba**: JIT compiler for optimizing Python functions for CUDA.
- **PyCUDA**: Library for integrating CUDA within Python.
- **NVIDIA CUDA**: Parallel computing platform and programming model.

## Report

For a detailed explanation of the parallelization strategies, performance results, and fine-tuning techniques, please refer to the [project report](A4_Report_55313.pdf).
