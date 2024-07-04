import numpy as np
import pandas as pd
from numba import cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as driver
import os
import time

def read_data(file_path):
    return pd.read_csv(file_path)

# Read the mathematical functions from a file and strip newline characters
def read_functions(file_path):
    return [line.strip() for line in open(file_path).readlines()]

# Function to convert a mathematical expression to a CUDA-compatible string
def convert_function_string(line, df_columns):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"{u[:-1]}")
    for c in df_columns:
        line = line.replace(f"_{c}_", f"df_{c}[i]")
    return line

# Build a CUDA evaluation block to evaluate each function for a given index.
# Each thread processes a different function, and the resulting values are accumulated in the results array
def gen_eval_block(functions):
    eval_block = ""
    for i, str_line in enumerate(functions):
        eval_block += f"""  if(idx == {i}){{
            for(int i = 0; i < num_items; i++){{
                double diff = {str_line} - df_y[i];
                double res_sqrt_diff = diff * diff;
                results[idx] += (!isnan(res_sqrt_diff)) ? res_sqrt_diff : 0.0;
                nan_count += isnan(res_sqrt_diff);
            }}
            results[idx] /= (num_items - nan_count);
        }}"""
    return eval_block


# Build a comma-separated string of DataFrame column pointers for the kernel
def build_eval_block(df_columns):
    return ", ".join([f"double *df_{c}" for c in df_columns if c != "Unnamed: 0"])

#  Compile the CUDA kernel using the generated code
def compile_cuda_kernel(eval_block, str_block):
    return SourceModule(f"""
    #include <math.h>
    #include <stdio.h>

    __global__ void eval(double *results, int num_items, int num_functions, {str_block}) {{
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int nan_count = 0;
        if(idx < num_functions){{
           {eval_block}
        }}
    }}
    """)

def main():
    # Set the CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda.select_device(0)

    # Record the start time for performance measurement
    start_time = time.time()

    df = read_data("data.csv")
    funs = read_functions("functions.txt")

    # Convert mathematical expressions to CUDA-compatible strings
    functions = [convert_function_string(line, df.columns) for line in funs]

    eval_block = gen_eval_block(functions)
    str_block = build_eval_block(df.columns)

    mod = compile_cuda_kernel(eval_block, str_block)

    # Get the compiled CUDA kernel function
    eval = mod.get_function("eval")

    num_items = len(df["a"])
    num_functions = len(funs)

    # Specify the number of threads per block and calculate the number of blocks
    THREADS_PER_BLOCK = 256
    BLOCKS_PER_GRID  = (len(funs) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Allocate device memory for the output and copy data to the device
    dev_output = driver.mem_alloc(np.zeros(len(funs), dtype=np.double).nbytes)

    # Device arrays for the dataframe cols
    dev_arrays_for_df = [cuda.to_device(df[c].values) for c in df.columns if c != "Unnamed: 0"]

    output = np.zeros(len(funs), dtype=np.double)

    # Execute the CUDA kernel
    eval(dev_output, np.int32(num_items), np.int32(num_functions), *dev_arrays_for_df,
         block=(THREADS_PER_BLOCK, 1, 1), grid=(BLOCKS_PER_GRID , 1, 1))
    
    print("THREADS_PER_BLOCK:", THREADS_PER_BLOCK)
    print("BLOCKS_PER_GRID:", BLOCKS_PER_GRID)
    # Copy the results back to the host
    driver.memcpy_dtoh(output, dev_output)

    # Find the index of the minimum score and corresponding function
    min_index = np.argmin(output)
    min_score = output[min_index]
    min_func = funs[min_index]

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Elapsed_time (Seconds): ", elapsed_time)
    print(f"Min Score: {min_score}, Function: {min_func}")

if __name__ == "__main__":
    main()
