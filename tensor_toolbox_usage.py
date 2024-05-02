# Filename: tensor_toolbox_usage.py

import numpy as np
import tensorflow as tf

# Import other functions
from getInnerProduct import getInnerProduct
from generateUniformField import generateUniformField
from getTrace import getTrace
from getEulerianTransformationMatrix import get_eulerian_transformation_matrix

def main():
    print("Step 1: Generating Uniform Vector Field...")
    input("Press Enter to continue...")
    
    # Generate a uniform vector field
    vec_field = generateUniformField("timelike", numAngularVec=10, numTimeVec=5, tryGPU=False)
    
    print("Uniform Vector Field Generated Successfully.")
    input("Press Enter to continue...")
    
    print("Step 2: Generating Metric Tensor...")
    input("Press Enter to continue...")
    
    # Generate a metric tensor
    metric_tensor = np.array([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, -1]])
    
    print("Metric Tensor Generated Successfully.")
    input("Press Enter to continue...")
    
    print("Step 3: Computing Trace of Metric Tensor...")
    input("Press Enter to continue...")
    
    # Compute the trace of the metric tensor
    metric_trace = getTrace(metric_tensor)
    
    print("Trace of Metric Tensor:", metric_trace)
    input("Press Enter to continue...")
    
    print("Step 4: Computing Eulerian Transformation Matrix...")
    input("Press Enter to continue...")
    
    # Compute the Eulerian transformation matrix
    eulerian_matrix = get_eulerian_transformation_matrix(metric_tensor)
    
    print("Eulerian Transformation Matrix Computed Successfully.")
    input("Press Enter to continue...")
    
    print("Step 5: Computing Inner Product of Vector Field...")
    input("Press Enter to continue...")
    
    # Compute the inner product of the vector field with itself using the metric tensor
    inner_product = getInnerProduct(vec_field, vec_field, metric_tensor)
    
    print("Inner Product Computed Successfully.")
    input("Press Enter to continue...")
    
    print("All Steps Completed Successfully. Exiting...")

if __name__ == "__main__":
    main()
