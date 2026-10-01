import numpy as np

@profile
def calculate_statistics(x):
    total = np.sum(x)
    mean = np.mean(x)
    variance = np.var(x)
    result = np.sqrt(variance)
    return total, mean, result

if __name__ == "__main__":
    x = np.random.normal(size=1_000_000)
    calculate_statistics(x)
    
    #OUTPUT WHEN RUN THROUGH TERMINAL
    """ Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     3                                           @profile
     4                                           def calculate_statistics(x):
     5         1       1262.4   1262.4     24.1      total = np.sum(x)
     6         1        650.6    650.6     12.4      mean = np.mean(x)
     7         1       3318.7   3318.7     63.3      variance = np.var(x)
     8         1          8.4      8.4      0.2      result = np.sqrt(variance)
     9         1          0.9      0.9      0.0      return total, mean, result """