import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp防止zscore过大
        # return np.round(your_answer, 4)
        maxz = np.max(z)
        f = np.exp(z-maxz)/np.sum(np.exp(z-maxz))
        return np.round(f,4)
