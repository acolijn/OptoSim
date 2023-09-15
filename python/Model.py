from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

class SuperResolutionModel:
    def __init__(self, low_to_high_res_net_params=None, high_res_to_true_net_params=None):
        """
        Initialize the SuperResolutionModel.

        Args:
            low_to_high_res_net_params (dict): Parameters for the low to high-resolution network.
            high_res_to_true_net_params (dict): Parameters for the high-resolution to true position network.
        """
        if low_to_high_res_net_params is None:
            low_to_high_res_net_params = {
                'hidden_layer_sizes': (100, 100),
                'max_iter': 500,
            }
        if high_res_to_true_net_params is None:
            high_res_to_true_net_params = {
                'hidden_layer_sizes': (100, 100),
                'max_iter': 500,
            }

        self.low_to_high_res_net = MLPRegressor(**low_to_high_res_net_params)
        self.high_res_to_true_net = MLPRegressor(**high_res_to_true_net_params)

    def create_datasets(self, X_low_res, X_high_res, y_true_pos, train_fraction=0.8):
        n_train = int(train_ratio*len(X))

        X_train = np.asarray(X_low_res[:n_train])
        y_train = np.asarray(X_high_res[:n_train])
        pos_train = np.asarray(y_true_pos[:n_train])

        X_test = np.asarray(X_low_res[n_train:])
        y_test = np.asarray(X_high_res[n_train:])
        pos_test = np.asarray(y_true_pos[n_train:])
        return X_train, y_train, pos_train, X_test, y_test, pos_test

    def train(self, X_low_res, X_high_res, y_true_pos):
        """
        Train the two networks.

        Args:
            X_low_res (array-like): Input data for low-resolution to high-resolution network.
            X_high_res (array-like): Input data for high-resolution to true position network.
            y_true_pos (array-like): Ground truth true position data.

        Returns:
            None
        """
        # Reshape the data
        X_low_res_flat = X_low_res.reshape(X_low_res.shape[0], -1)
        X_high_res_flat = X_high_res.reshape(X_high_res.shape[0], -1)
        y_true_pos_flat = y_true_pos.reshape(y_true_pos.shape[0], -1)

        # Train the low to high-resolution network
        self.low_to_high_res_net.fit(X_low_res_flat, X_high_res_flat)

        # Train the high-resolution to true position network
        self.high_res_to_true_net.fit(X_high_res_flat, y_true_pos_flat)

    def predict(self, X_low_res):
        """
        Predict the true position from a low-resolution input.

        Args:
            X_low_res (array-like): Input data for low-resolution to high-resolution network.

        Returns:
            array-like: Predicted true positions.
        """
        # Reshape the data
        X_low_res_flat = X_low_res.reshape(X_low_res.shape[0], -1)

        # Upscale from low resolution to high resolution
        X_high_res_pred = self.low_to_high_res_net.predict(X_low_res_flat)

        # Predict true position from high resolution
        y_true_pos_pred = self.high_res_to_true_net.predict(X_high_res_pred)

        return y_true_pos_pred

    def evaluate(self, X_low_res, X_high_res, y_true_pos):
        """
        Evaluate the model on a test dataset.

        Args:
            X_low_res (array-like): Input data for low-resolution to high-resolution network.
            X_high_res (array-like): Input data for high-resolution to true position network.
            y_true_pos (array-like): Ground truth true position data.

        Returns:
            dict: Evaluation metrics (e.g., MSE, R^2).
        """
        # Reshape the data
        X_low_res_flat = X_low_res.reshape(X_low_res.shape[0], -1)
        X_high_res_flat = X_high_res.reshape(X_high_res.shape[0], -1)
        y_true_pos_flat = y_true_pos.reshape(y_true_pos.shape[0], -1)

        y_pred = self.predict(X_low_res_flat)

        mse = np.mean((y_true_pos_flat - y_pred) ** 2)
        r_squared = 1 - (np.sum((y_true_pos_flat - y_pred) ** 2) / np.sum((y_true_pos_flat - np.mean(y_true_pos_flat)) ** 2))

        return {'MSE': mse, 'R^2': r_squared}

    def plot_loss_curve(self):
        """
        Plot the training loss curve.
        """
        if hasattr(self.low_to_high_res_net, 'loss_curve_'):
            plt.figure(figsize=(10, 5))
            plt.plot(self.low_to_high_res_net.loss_curve_, label='Low to High Res Loss', color='blue')
            plt.plot(self.high_res_to_true_net.loss_curve_, label='High Res to True Loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training Loss Curves')
            plt.show()
        else:
            print("Loss curves not available. Ensure you have trained the models.")

    import matplotlib.pyplot as plt

    def visualize_heatmaps_with_positions(self, X_low_res, X_high_res, y_true_pos):
        """
        Visualize input and output heatmaps along with true and predicted positions.

        Args:
            X_low_res (array-like): Input data for low-resolution to high-resolution network.
            y_true_pos (array-like): Ground truth true position data.

        Returns:
            None
        """
        original_shape = X_high_res.shape
        # Reshape the data
        X_low_res_flat = X_low_res.reshape(X_low_res.shape[0], -1)
        X_high_res_flat = X_high_res.reshape(X_high_res.shape[0], -1)
        y_true_pos_flat = y_true_pos.reshape(y_true_pos.shape[0], -1)
        
        # Predict high-resolution heatmaps
        X_high_res_pred = self.low_to_high_res_net.predict(X_low_res_flat)

        # Predict true positions
        y_pred_pos = self.high_res_to_true_net.predict(X_high_res_pred)

        num_samples = 10#len(X_low_res)

        for i in range(num_samples):
            plt.figure(figsize=(12, 4))

            # Plot input low-resolution heatmap
            plt.subplot(1, 3, 1)
            plt.imshow(X_low_res[i], cmap='hot', interpolation='nearest', origin='lower', extent=[-2.5, 2.5, -2.5, 2.5])
            plt.scatter(y_true_pos[i, 0], y_true_pos[i, 1], c='r', label='True Position')
            plt.scatter(y_pred_pos[i, 0], y_pred_pos[i, 1], c='b', label='Predicted Position')
            plt.title('Input Low-Res Heatmap')

            # Plot predicted high-resolution heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(X_high_res_pred[i].reshape(original_shape[1:3]), cmap='hot', interpolation='nearest', origin='lower', extent=[-2.5, 2.5, -2.5, 2.5])
            plt.scatter(y_true_pos[i, 0], y_true_pos[i, 1], c='r', label='True Position')
            plt.scatter(y_pred_pos[i, 0], y_pred_pos[i, 1], c='b', label='Predicted Position')
            plt.title('Predicted High-Res Heatmap')

            # Plot true and predicted positions
            #plt.subplot(1, 3, 3)
            #plt.imshow(X_high_res_pred[i].reshape(20,20), cmap='hot', interpolation='nearest')
            #plt.scatter(y_true_pos[i, 0], y_true_pos[i, 1], c='r', label='True Position')
            #plt.scatter(y_pred_pos[i, 0], y_pred_pos[i, 1], c='b', label='Predicted Position')
            plt.legend()
            plt.title('True vs. Predicted Position')

            plt.show()

    def visualize_x_vs_true_x(self, X_low_res, y_true_pos):
        """
        Visualize predicted x-coordinate vs. true x-coordinate.

        Args:
            X_low_res (array-like): Input data for low-resolution to high-resolution network.
            y_true_pos (array-like): Ground truth true position data.

        Returns:
            None
        """
        original_shape = X_low_res.shape
        # Reshape the data
        X_low_res_flat = X_low_res.reshape(X_low_res.shape[0], -1)

        # Predict high-resolution heatmaps
        X_high_res_pred = self.low_to_high_res_net.predict(X_low_res_flat)

        # Predict true positions
        y_pred_pos = self.high_res_to_true_net.predict(X_high_res_pred)

        # Extract x-coordinates from true and predicted positions
        true_x = y_true_pos[:, 0]
        true_y = y_true_pos[:, 1]
        predicted_x = y_pred_pos[:, 0]
        predicted_y = y_pred_pos[:, 1]

        # Create a scatter plot of predicted x vs. true x
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(true_x, predicted_x, c='b', label='Predicted vs. True x')
        plt.xlabel('True x-coordinate')
        plt.ylabel('Predicted x-coordinate')
        plt.title('Predicted x vs. True x')
        plt.grid(True)
        plt.legend() 

        plt.subplot(1, 2, 2)
        plt.scatter(true_y, predicted_y, c='b', label='Predicted vs. True y')
        plt.xlabel('True y-coordinate')
        plt.ylabel('Predicted y-coordinate')
        plt.title('Predicted y vs. True y')
        plt.grid(True)
        plt.legend()

        plt.show()

def create_datasets(X_low_res, X_high_res, y_true_pos, train_fraction=0.8):
    n_train = int(train_fraction*len(X_low_res))

    X_train = np.asarray(X_low_res[:n_train])
    y_train = np.asarray(X_high_res[:n_train])
    pos_train = np.asarray(y_true_pos[:n_train])

    X_test = np.asarray(X_low_res[n_train:])
    y_test = np.asarray(X_high_res[n_train:])
    pos_test = np.asarray(y_true_pos[n_train:])
    return X_train, y_train, pos_train, X_test, y_test, pos_test