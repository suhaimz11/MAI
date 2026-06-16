import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Create synthetic dataset
# -------------------------------

np.random.seed(42)  # for reproducibility

# Generate 500 age values (normally distributed)
ages = np.random.normal(loc=10, scale=3, size=500)

# Ensure realistic age range (e.g., 5 to 18 years)
ages = np.clip(ages, 5, 18)

# Create DataFrame
df = pd.DataFrame({"age": ages})

# -------------------------------
# Step 2: Convert to integer ages
# -------------------------------

# Round ages to nearest integer
ages_int = df["age"].round().astype(int)

# -------------------------------
# Step 3: Count frequencies
# -------------------------------

counts = ages_int.value_counts().sort_index()

# -------------------------------
# Step 4: Convert to probabilities
# -------------------------------

probabilities = counts / counts.sum()

# -------------------------------
# Output results
# -------------------------------

print("Counts:\n", counts)
print("\nProbabilities:\n", probabilities)



# -------------------------------
# Exercise 1(b): Sampling
# -------------------------------

import random

# Step 1: Extract values and probabilities
values = probabilities.index.tolist()
probs = probabilities.values.tolist()

# Step 2: Create cumulative distribution
cumulative = np.cumsum(probs)

# Step 3: Sampling function
def sample_multinomial():
    u = random.random()  # random number in [0,1]
    
    for i, c in enumerate(cumulative):
        if u <= c:
            return values[i]

# Step 4: Generate samples
samples = [sample_multinomial() for _ in range(1000)]

print("\nSampled values (first 20):\n", samples[:20])


#exercise 1c
sample_counts = pd.Series(samples).value_counts().sort_index()

# Convert to probabilities
sample_probs = sample_counts / sample_counts.sum()


import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(8,5))

# Plot original distribution
plt.plot(probabilities.index, probabilities.values, marker='o', label="Original")

# Plot sampled distribution
plt.plot(sample_probs.index, sample_probs.values, marker='x', label="Sampled")

# Labels and title
plt.xlabel("Age")
plt.ylabel("Probability")
plt.title("Original vs Sampled Distribution")

# Legend
plt.legend()

# Show plot
plt.show()