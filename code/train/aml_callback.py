import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf


class AzureMLCallback(tf.keras.callbacks.Callback):
    def __init__(self, aml_logger):
        self.run = aml_logger
        self.history = {}
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        for name, value in logs.items():
            self.run.log(name, value)

            if self.history.get(name):
                self.history[name].append(value)
            else:
                self.history[name] = [value]

    def on_train_end(self, logs=None):
        self.run.log_image("accuracy_by_epoch",
                           plot=self.__generate_plot("acc"))
        self.run.log_image("loss_by_epoch", plot=self.__generate_plot("loss"))

    def __generate_plot(self, metric):
        """Generate a plot based on history and """
        fig, ax = plt.subplots()

        dt = pd.DataFrame(
            {k: v for k, v in self.history.items() if metric in k})
        sns.lineplot(data=dt, legend="brief")
        plt.close(fig)

        return fig
