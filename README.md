## Coord-Prediction

### Description
This project predicts whether 2D points are inside or outside a circle defined by a center and radius. It is built around a neural network implemented from scratch.

---

### Project Structure

**Main Files:**
- `neural_network.py`: Contains the `NeuralNetwork` class, backpropagation implementation, and test methods.
- `coord_pred.py`: Generates the dataset, initializes and trains the neural network, and runs tests.

---

### Usage Guide

#### Step 1: Data Generation
1. The `coord_pred.py` script generates a dataset of random points (x, y).
2. Each point is labeled based on whether it is inside or outside a defined circle.
3. A visualization of the training dataset is saved as `TRAIN_points.png`.

#### Step 2: Training
1. The neural network is configured with the following parameters:
   - 2 input neurons (x, y),
   - n hidden neurons (works with 4),
   - 1 output neuron.
   - Activation function: Sigmoid. (you can choose Relu also)
   - Learning rate: 0.5 for Sigmoid (you can change it).
2. Training runs for 1000 epochs with a batch size of 1 (you can modifiy both).
3. The `neural_network.py` script prints the Mean Squared Error (MSE) at regular intervals.

#### Step 3: Results and Testing
1. After training, `coord_pred.py` runs the network on test data.
2. A plot of the results is saved as `PREDICTION_points.png`, where:
   - Points inside the circle are green.
   - Points outside the circle are red.

---

### Customizable Parameters
In `coord_pred.py`:
- **Circle center**: Defined by `circle_center`.
- **Circle radius**: Set using the `circle_radius` variable.
- **Number of generated points**: Controlled by `n_points`.
- **Learning rate**: Configured in the `NeuralNetwork` class.

