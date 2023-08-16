import pandas as pd
import string
import random
import time
import datetime
import numpy as np
import pickle


def get_patient_ids(num_data: int) -> list:
    """Generate random list of patient ids

    Args:
        num_data (int): number of ids to generate

    Returns:
        list: generated patient ids
    """
    patient_ids = []
    for p in range(num_data):
        patient_ids.extend(
            ["".join(random.choice(string.ascii_lowercase) for i in range(6))]
        )
    return patient_ids


def str_time_prop(
    start: str, end: str, time_format: str = "%m/%d/%Y", prop: float = 0.5
) -> datetime.datetime:
    """
    https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates

    Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be a datetime object.
    """

    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)

    delta = etime - stime
    offset = delta * prop

    return stime + offset


def random_date(start: str = "1920-01-01", end: str = "2020-01-01", prop=None):
    """
    return date at location prop (decimal between 0 and 1) between dates start and end
    """
    if prop is None:
        prop = random.uniform(0, 1)
    return str_time_prop(start, end, "%Y-%m-%d", prop)


def insert_nans(lst, pct_nan):
    """
    Inserts NaNs randomly into a list.

    Args:
        lst (list): List to insert NaNs into.
        pct_nan (float): Percentage of NaNs to insert, between 0 and 1.

    Returns:
        list: List with NaNs inserted.
    """
    nan_indices = random.sample(range(len(lst)), int(len(lst) * pct_nan))
    return [np.nan if i in nan_indices else x for i, x in enumerate(lst)]
