import os
import os.path as op
from apps.svd.TabData import TabData
import pandas as pd
import traceback
import numpy as np
import copy
from apps.svd.algo_params import QR
from apps.svd.SVD import SVD
from apps.svd.params import INPUT_DIR, OUTPUT_DIR
from apps.svd.COParams import COParams
import time
import scipy.linalg as la
import scipy.sparse.linalg as lsa
import apps.svd.shared_functions as sh
from apps.svd.params import INPUT_DIR, OUTPUT_DIR
import shutil

class FCFederatedMFA:
    def __init__(self):
        self.number_of_omics = 0