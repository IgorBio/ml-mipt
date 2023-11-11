import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    n = data.shape[0]

    eigenvector = np.random.rand(n)

    for _ in range(num_steps):
        eigenvector = data.dot(eigenvector)
        eigenvector /= np.linalg.norm(eigenvector)

    eigenvalue = float(eigenvector.dot(data.dot(eigenvector)))
    return eigenvalue, eigenvector