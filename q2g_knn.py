import numpy as np
from q2e_word2vec import normalizeRows


def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """
    ### YOUR CODE
    """ Empty matrix """
    if len(matrix.shape)==0:
        return np.array([])

    """ More than one row checking if we have at least 10 otherwise setting to new value"""
    if len(matrix.shape)>1:
        k = np.minimum(k,matrix.shape[1])
    else:
        return np.array([0])

    # cos similarity between 2 vectors
    # sigma (Ai dot Bi) / (norm(A) * norm(B))
    # norm(A) = sqrt(sigma(Ai) ** 2)

    # enumerator V dot matrix(i) 
    dotMatrix  = np.dot(matrix,vector)
    # denominator norm(matrix(i))
    normalMatrix  =  np.apply_along_axis(lambda x: np.linalg.norm(x), 1,matrix)
    print normalMatrix
    normalVector = np.linalg.norm(vector)
    # denom norm(V) * norm(matrix(i))
    normalsProduct = normalVector*normalMatrix
    print normalsProduct
    # V dot matrix(i) / norm(V) * norm(matrix(i))
    cosValues = dotMatrix/normalsProduct
    # increasing order: return the indices
    indicesflipped =  np.argsort(cosValues)
    # flip in the left/right direction
    # remove one dimension
    indices = np.fliplr([indicesflipped])[0]
    nearest_idx  = indices[0:k]
    ### END YOUR CODE
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
        your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    indices = knn(np.array([0.2,0.5]), np.array([[0,0.5],[0.1,0.1],[0,0.5],[2,2],[4,4],[3,3]]), k=2)
    print indices
    assert 0 in indices and 2 in indices and len(indices) == 2

    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


