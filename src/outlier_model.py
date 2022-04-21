import collections
import logging

import numpy as np
import stumpy

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger("TimeSeries")
logger.setLevel(logging.INFO)


class OutlierModel:
    """Object used to define outlier network designs"""

    def __init__(
            self,
            m: int = 15,
            std_dev_mult: float = 3.0,
            range_mult: float = 3.0,
            recent_mult: float = 2.0,
            time_series=None,
            egress=True
    ):
        self.m = m
        if time_series is None:
            time_series = np.zeros(self.m * 4)
        self.time_series = time_series
        self.ts_size = len(self.time_series)
        self.stream = stumpy.stumpi(self.time_series, m, egress=egress, normalize=False)
        self.anomalies = []
        self.std_dev_mult = std_dev_mult
        self.range_mult = range_mult
        self.recent_mult = recent_mult
        self.count = 0
        self.max_std_dev = []
        self.max_mean = []
        self.max_val = []
        self.stdFilter = []
        self.rolling_range = collections.deque(np.zeros(5), maxlen=5)

    def __repr__(self):
        return str(self.__dict__)

    def train(self):
        pass

    def train_one(self, data):
        self.stream.update(data)

    def predict_one(self, index):
        self.count += 1
        max_mp = np.round(self.stream.P_.max(), 4)
        mean_mp = np.round(self.stream.P_.mean(), 4)
        std_dev_mp = np.round(self.stream.P_.std(), 4)
        self.max_val.append(max_mp)
        self.max_mean.append(mean_mp)
        self.max_std_dev.append(std_dev_mp)

        true_anomaly = False

        metric = (mean_mp + (std_dev_mp * self.std_dev_mult))
        self.stdFilter.append(metric)

        if self.filter(index) and (max_mp > metric) and (abs(metric - max_mp) > 0.01) and self.range_filter():
            self.rolling_range.append(self.range())
            true_anomaly = True
            self.anomalies.append(index)
            logger.info(f" Anomaly at Global index: {index}")
            logger.info(f"max_mp: {max_mp}, metric:{metric}: metric-max_mp: {abs(metric - max_mp)} range: {self.range()}")
        return true_anomaly

    def filter(self, index):
        return (not self.is_warming_up()) and (not self.recent_fault(index))

    def range(self):
        window = self.max_val[-self.m:]
        return max(window) - min(window)

    def range_filter(self):
        return self.range() > (self.range_mult * np.average(self.rolling_range))

    def is_warming_up(self):
        return self.count <= self.m

    def recent_fault(self, index):
        return index < (
                self.anomalies[-1] + self.ts_size * self.recent_mult) if self.anomalies else False
