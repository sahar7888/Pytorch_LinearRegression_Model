
""" This class plots train vs test data"""

import matplotlib.pyplot as plt

class PLOT:
    def __init__(self, train_data,
                 train_label,
                 test_data,
                 test_label):
        self.train_data = train_data
        self.train_label=train_label
        self.test_data=test_data
        self.test_label=test_label

    def plot_predictions(self, predictions=None):
        """
        Plots training data, test data and compares predictions.
        """
        plt.figure(figsize=(10, 7))

        # Plot training data in blue
        plt.scatter(self.train_data, self.train_label, c="b", s=4, label="Training data")

        # Plot test data in green
        plt.scatter(self.test_data, self.test_label, c="g", s=4, label="Testing data")

        if predictions is not None:
            # Plot the predictions in red (predictions were made on the test data)
            plt.scatter(self.test_data, predictions, c="r", s=4, label="Predictions")

        # Show the legend
        plt.legend(prop={"size": 14})

        plt.show()
