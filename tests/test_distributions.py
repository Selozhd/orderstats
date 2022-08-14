import unittest
import pytest

import numpy as np
from numpy import random
from scipy import stats
from orderstats import distributions


class Test_DistributionFunction(unittest.TestCase):
    
    def setUp(self):
        self.n = 1000
        self.m = random.randint(self.n)