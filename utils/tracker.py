""" 
Tracker class to track metrics during training and occasional evaluation on validation set
"""
import pickle
import pandas as pd
from collections import defaultdict

class Tracker:
    """
    Tracker mother class

    ...

    Attributes
    ----------
    name (str): Name of the tracker
    labels (list[str]): The labels for the classes that were trained on (0, 1)
    metrics (defaultdict): A defaultdict of (key, value) pairs that store metric data.
    """

    def __init__(self, name: str, labels: list[str]):
        """

        Args:
            name (str): Name of the tracker
            labels (list[str]): The labels for the classes that were trained on (a-z, A-Z or A-z)
        """
        super().__init__()
        self.name = name
        self.metrics = defaultdict(list)
        self.labels = labels
        
    def save_data_dict(self, path: str):
        """Saves the tracking data for later use as a .pkl.

        Args:
            path (str): path/to/save/tracker
        """
        last_epoch = self.metrics["epoch"][len(self.metrics["epoch"]) - 1]
        with open(
            path + "/" + self.name + "_" + str(last_epoch) + ".pkl", "wb"
        ) as pkl_handle:
            pickle.dump(self.metrics, pkl_handle)

    def read_data_dict(self, file_path: str):
        """Read a data dict that was saved with save_data_dict().

        Args:
            file_path (str): path/to/dict.pkl
        """
        with open(file_path, "rb") as pkl_handle:
            self.metrics = pickle.load(pkl_handle)

    def save_1d_metrics_xlsx(self, file_path: str):
        """Saves the one-dimensional metrics (i.e., all except the aleatoric/epistemic matrices) as an .xlsx

        Args:
            file_path (str): path/to/excelfile.xlsx (...plotter/testmetrics.xlsx)
        """
        df = pd.DataFrame.from_dict(dict([item for item in self.metrics.items()][2:18]))
        df.to_excel(file_path)

class TrainTracker(Tracker):
    def update_epoch(self, epoch, metrics: dict, time):
        """Updates the metrics over training time. Only most important metrics are saved.

        Args:
            epoch: The epoch to record
            mean_loss: cross entropy loss
            mean_dice: dice coefficient loss
            mean_accuracy: The accuracy
            time: The time it took for training.
        """
        for key, value in [
            ("epoch", epoch),
            ("mean_loss", mean_loss),
            ("mean_dice", mean_dice),
            ("accuracy", mean_accuracy),
            ("time", time),
        ]:
            self.metrics[key].append(value)
