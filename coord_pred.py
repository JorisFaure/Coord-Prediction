import numpy as np
import time
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# Data creation
circle_center = (0.5, 0.5)
circle_radius = 0.3
num_points = 100 # Number of points to generate

x = np.random.rand(num_points)
y = np.random.rand(num_points)

distances = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)
labels = np.where(distances <= circle_radius, 1, 0) # 1 (blue) if inside the circle, 0 (red) otherwise

input_set = np.column_stack((x, y))  #Create coordonates
output_set = labels.reshape(-1, 1)    #put in form [[1],[0],[1]...]

#We plot the points
plt.figure(figsize=(8, 8))
plt.scatter(x[labels == 1], y[labels == 1], color='blue', label='Class 1')
plt.scatter(x[labels == 0], y[labels == 0], color='red', label='Class 0')

circle = plt.Circle(circle_center, circle_radius, color='black', fill=False, linestyle='--', linewidth=2) #circle for a better understanding of plotting zone
plt.gca().add_artist(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("TRAIN_points.png")
print("Plot saved as 'TRAIN_points.png'")


nn = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, activation_function='sigmoid')

# Mesure training time
print("Starting training...")
start_time = time.time()
nn.train(input_set, output_set, epochs=1000, batch_size=1)
end_time = time.time()
print("Training over")

training_time = end_time - start_time
print(f"Training time : {training_time:.2f} seconds.")

# Network test
print("\nTest result :")
nn.plot_test(input_set, output_set)