import math

# Ensure the probabilities are positive before taking the log
def safe_log(x):
    if x > 0:
        return math.log(x)
    else:
        # Handle the case where probability is zero or negative
        return 1e-10
