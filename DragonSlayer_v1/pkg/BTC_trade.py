# -*- coding: utf-8 -*-
"""
BTC-USDT info updater

"""

from datetime import datetime, timedelta

import numpy as np
np.float_ = np.float64

import ccxt
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingClassifier

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


# Training

def signal():
    
    