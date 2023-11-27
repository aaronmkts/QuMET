import numpy as np
import matplotlib.pyplot as plt 
'''
def generate_samples(n, mean, std_devs, probabilities, num_samples):
    # Check if the input parameters are valid
    if n != len(mean) or n != len(probabilities):
        raise ValueError("Number of standard deviations and probabilities should match the number of distributions.")

    # Check if probabilities add up to 1
    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities should add up to 1.")

    samples = []

    for _ in range(num_samples):
        # Choose a distribution based on probabilities
        chosen_distribution = np.random.choice(np.arange(n), p=probabilities)

        # Generate a random sample from the chosen distribution
        sample = np.random.normal(mean[chosen_distribution], std_dev)

        samples.append(sample)

    return samples

# Example usage for 1000 samples
n = 5  # Number of Gaussian distributions
mean = [1, 2, 3, 4, 5]  # Mean for all distributions

# Specify different standard deviations and probabilities
std_dev = 0.1
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]

# Generate 1000 random samples
num_samples = 1000
random_samples = generate_samples(n, mean, std_dev, probabilities, num_samples)

# Print the first few samples
print("First 5 samples:", random_samples[:5])

plt.hist(random_samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.show()

'''