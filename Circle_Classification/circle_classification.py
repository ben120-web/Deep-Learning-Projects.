from sklearn.datasets import make_circles

# Make 1000 samples
n_samples = 1000

# Create circles, with some noise
X, y = make_circles(n_samples, noise = 0.03,
                    random_state = 42)

# View the first 5 X and y values.
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

# Make dataframe of circle data.
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

circles.head(10)

# Check the different labels.
circles.label.value_counts()

# Visualise the circles.
import matplotlib.pyplot as plt
plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y, 
            cmap = plt.cm.RdY1Bu)

# Check the shapes of our features and labels
X.shape, y.shape

# View the first example of features and labels.
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of x: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

## Turn data to tensores and create train and test split.

                
