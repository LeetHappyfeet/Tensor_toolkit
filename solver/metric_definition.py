# Define the metric tensor dictionary with a 'scaling' key
metric = {
    'type': 'Metric',  
    'tensor': [
        [[1, 0, 0, 0],   
         [0, -1, 0, 0],  
         [0, 0, -1, 0],  
         [0, 0, 0, -1]], 
    ],
    'coords': 'cartesian',   
    'index': 'covariant',   
    'scaling': 1.0  # Adding the scaling factor
}
