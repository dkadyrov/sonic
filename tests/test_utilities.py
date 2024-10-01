import os

import pytest
import numpy as np
import pandas as pd
from datetime import timedelta
from sonicdb import sonicdb, models, audio, utilities


# @pytest.fixture(scope="module")
def test_lower_keys():
    d = {"A": 1, "B": 2}
    d = utilities.lower_keys(d)

    assert d == {"a": 1, "b": 2}
