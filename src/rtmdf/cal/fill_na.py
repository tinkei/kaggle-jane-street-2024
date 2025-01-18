"""Unused."""

from enum import Enum, auto


class FillStrategy(Enum):
    ZERO = auto()
    MEAN = auto()
    MEDIAN = auto()
    GROUP_MEAN = auto()
    GROUP_MEDIAN = auto()


FILL_STRATEGY = {
    "feature_00": FillStrategy.MEAN,
    "feature_01": FillStrategy.ZERO,
    "feature_02": FillStrategy.MEAN,
    "feature_03": FillStrategy.MEAN,
    "feature_04": FillStrategy.ZERO,
    "feature_05": FillStrategy.ZERO,
    "feature_06": FillStrategy.ZERO,
    "feature_07": FillStrategy.MEAN,
    "feature_08": FillStrategy.MEAN,
    "feature_09": FillStrategy.MEAN,  # Looks int
    "feature_10": FillStrategy.MEAN,  # Looks int
    "feature_11": FillStrategy.MEAN,  # Looks int
    "feature_12": FillStrategy.ZERO,
    "feature_13": FillStrategy.ZERO,
    "feature_14": FillStrategy.ZERO,
    "feature_15": FillStrategy.MEAN,
    "feature_16": FillStrategy.MEAN,
    "feature_17": FillStrategy.MEAN,
    "feature_18": FillStrategy.ZERO,
    "feature_19": FillStrategy.ZERO,
    "feature_20": FillStrategy.MEAN,  # Looks const
    "feature_21": FillStrategy.MEAN,  # Looks const
    "feature_20": FillStrategy.MEAN,  # Looks const
    "feature_23": FillStrategy.MEAN,  # Looks const
    "feature_24": FillStrategy.MEAN,  # Looks const
    "feature_25": FillStrategy.MEAN,  # Looks const
    "feature_26": FillStrategy.MEAN,  # Looks const
    "feature_27": FillStrategy.MEAN,  # Looks const
    "feature_28": FillStrategy.MEAN,  # Looks const
    "feature_29": FillStrategy.MEAN,  # Looks const
    "feature_30": FillStrategy.MEAN,  # Looks const
    "feature_31": FillStrategy.MEAN,  # Looks const
    "feature_32": FillStrategy.MEAN,
    "feature_33": FillStrategy.MEAN,
    "feature_34": FillStrategy.MEAN,
    "feature_35": FillStrategy.MEAN,
    "feature_36": FillStrategy.ZERO,
    "feature_37": FillStrategy.MEAN,
    "feature_38": FillStrategy.MEAN,
    "feature_39": FillStrategy.ZERO,
}
