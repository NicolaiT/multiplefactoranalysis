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
        # self.step = 0
        self.tabdatas = []
        # self.pca = None
        self.config_available = False
        self.outs = []
        self.send_data = False
        self.computation_done = False
        self.coordinator = False
        
        # self.step_queue = [] # this is the initial step queue
        # self.state = 'waiting_for_start' # this is the inital state
        # self.iteration_counter = 0
        # self.converged = False
        # self.outliers = []
        # self.approximate_pca = True
        # self.data_incoming = {}
        # self.progress = 0.0
        # self.silent_step=False
        # self.use_smpc = False
        # self.start_time = time.monotonic()
        # self.pre_iterations = 10


        self.means = []
        self.std = []
        self.sos = []
        self.variances = []

        # self.total_sampels = 0
        self.total_sampels = []
    
    def get_configuration(self):
        config = {
            'exponent': self.exponent,
            'sep': self.sep,
            'has_rownames': self.has_rownames,
            'has_colnames': self.has_colnames,
            'subsample': self.subsample,
            'center': self.center,
            'unit_variance': self.unit_variance,
            'highly_variable': self.highly_variable,
            'perc_highly_var': self.perc_highly_var,
            'log_transform': self.log_transform,
            'max_nan_fraction': self.max_nan_fraction
        }
        return config
    
    def get_omics_data(self, idx):
        omics_data = {
            'k': self.k[idx],
            'k2': self.k2[idx],
            'tabdata': self.tabdatas[idx],
            'means': self.means[idx],
            'std': self.std[idx],
            'sos': self.sos[idx],
            'variances': self.variances[idx],
            'total_sampels': self.total_sampels[idx]
        }
        return omics_data
    
    def copy_configuration(self, config):
        print('[STARTUP] Copy configuration and create dir')
        self.config_available = config.config_available        
        self.input_files = [op.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if file.startswith(config.input_files) and file.endswith(config.extension)]
        
        os.makedirs(op.join(OUTPUT_DIR, config.output_dir), exist_ok=True)
        self.left_eigenvector_file = op.join(OUTPUT_DIR,config.output_dir,  config.left_eigenvector_file)
        self.right_eigenvector_file = op.join(OUTPUT_DIR,config.output_dir, config.right_eigenvector_file)
        self.eigenvalue_file = op.join(OUTPUT_DIR,config.output_dir, config.eigenvalue_file)
        self.projection_file = op.join(OUTPUT_DIR,config.output_dir, config.projection_file)
        self.scaled_data_file =  op.join(OUTPUT_DIR,config.output_dir, config.scaled_data_file)
        self.explained_variance_file = op.join(OUTPUT_DIR,config.output_dir, config.explained_variance_file)
        self.means_file = op.join(OUTPUT_DIR,config.output_dir, 'mean.tsv')
        self.stds_file = op.join(OUTPUT_DIR,config.output_dir, 'std.tsv')
        self.log_file = op.join(OUTPUT_DIR,config.output_dir, 'run_log.txt')
        self.output_delimiter = config.output_delimiter
        
        self.k = [config.k] * len(self.input_files)

        # self.k = config.k

        self.exponent = config.exponent

        self.sep = config.sep
        self.has_rownames = config.has_rownames
        self.has_colnames = config.has_colnames
        self.send_projections = config.send_projections
        self.subsample = config.subsample

        self.center = config.center
        self.unit_variance = config.unit_variance
        self.highly_variable = config.highly_variable
        self.perc_highly_var = config.perc_highly_var
        self.log_transform = config.log_transform
        self.max_nan_fraction = config.max_nan_fraction
        
        
    def read_input_file(self, input_file):
        # self.progress = 0.1
        tabdata = TabData.from_file(input_file, header=self.has_colnames,
                                         index=self.has_rownames, sep=self.sep)

        if self.log_transform:
            print('Log Transform performed')
            tabdata.scaled = np.log2(tabdata.scaled+1)
            print('tabdata.scaled', tabdata.scaled)

        nans = np.sum(np.isnan(tabdata.scaled), axis=1)
        infs = np.sum(np.isinf(tabdata.scaled), axis=1)
        isneginf = np.sum(np.isneginf(tabdata.scaled), axis=1)
        nans = np.sum([nans, isneginf, infs], axis=0)
        out = {COParams.ROW_NAMES.n : tabdata.rows, COParams.SAMPLE_COUNT.n: tabdata.col_count, COParams.NAN.n: nans}
        
        return tabdata, out

    def read_input_files(self):
        for file in self.input_files:
            tabdata, out = self.read_input_file(file)
            self.tabdatas.append(tabdata)
            self.outs.append(out)
            
    def set_parameters(self, incomings):
        self.k2 = [0] * len(self.input_files)
        print("INCOMINGS")
        print(incomings)
        for idx, incoming in enumerate(incomings):
            print("INCOMING")
            print(incoming)
            print("K")
            print(self.k)
            print("incoming[COParams.PCS.n]")
            print(incoming[COParams.PCS.n]) 
            self.k[idx] = incoming[COParams.PCS.n]
            self.k2[idx] = incoming[COParams.PCS.n]*2

    
    def select_rows(self, incomings):
        for idx, incoming in enumerate(incomings):
            subset = incoming[COParams.ROW_NAMES.n]
            print(subset)
            d = {k: v for v, k in enumerate(self.tabdatas[idx].rows)}
            index = []
            for elem in subset:
                if elem in d:
                    index.append(d[elem])
            print('INDEX')
            print(index)
            self.tabdatas[idx].scaled = self.tabdatas[idx].scaled[index,:]
            self.tabdatas[idx].rows = self.tabdatas[idx].rows[index]
            self.tabdatas[idx].row_count = len(self.tabdatas[idx].rows)  
    
    def compute_sums(self):
        self.sums = []
        for idx, tabdata in enumerate(self.tabdatas):
            self.sums.append(np.nansum(tabdata.scaled, axis=1))

            self.outs[idx] = {COParams.SUMS.n: self.sums[idx], COParams.SAMPLE_COUNT.n: tabdata.col_count}      
                
    def compute_sum_of_squares(self, incomings):
        for idx, incoming in enumerate(incomings):
            self.means.append(incoming[COParams.MEANS.n].reshape((len(incoming[COParams.MEANS.n]),1)))
            print(self.means[idx].shape)
            self.sos.append(np.nansum(np.square(self.tabdatas[idx].scaled-self.means[idx]), axis=1))
            self.outs[idx] = {COParams.SUM_OF_SQUARES.n: self.sos[idx].flatten()}            
     
    def apply_scaling(self, incomings, highly_variable=True):
        output = []
        for idx, incoming in enumerate(incomings):
            self.std.append(incoming[COParams.STDS.n][idx].reshape((len(incoming[COParams.STDS.n][idx]),1)))
            self.variances.append(incoming[COParams.VARIANCES.n][idx])
            remove = incoming[COParams.REMOVE.n] # remove due to 0
            select = incoming[COParams.SELECT.n] # select due to highly var

            # for row in range(self.tabdata.scaled.shape[0]):
            #     self.tabdata.scaled[row, :]= self.tabdata.scaled[row, :]- self.means[row,0]
            if self.center:
                self.tabdatas[idx].scaled = np.subtract(self.tabdatas[idx].scaled,self.means[idx])


            # self.tabdata.scaled = np.delete(self.tabdata.scaled, remove)
            # self.tabdata.rows = np.delete(self.tabdata.rows, remove)
            print("self.tabdatas[idx].scaled")
            print(self.tabdatas[idx].scaled)
            print("self.std[idx]")
            print(self.std[idx])
            print("self.std")
            print(self.std)
            if self.unit_variance:
                self.tabdatas[idx].scaled = self.tabdatas[idx].scaled/self.std[idx]

            if self.center:
                # impute. After centering, the mean should be 0, so this effectively mean imputation
                self.tabdatas[idx].scaled = np.nan_to_num(self.tabdatas[idx].scaled, nan=0, posinf=0, neginf=0)
            else:
                # impute
                self.tabdatas[idx].scaled = np.where(np.isnan(self.tabdatas[idx].scaled), self.means[idx], self.tabdatas[idx].scaled)

            print(select)
            print(remove)
            if highly_variable:
                self.tabdatas[idx].scaled = self.tabdatas[idx].scaled[select, :]
                self.tabdatas[idx].rows = self.tabdatas[idx].rows[select]
                print('Selected')
            output = self.tabdatas[idx].scaled.shape[0] 
        return output
     
    def save_scaled_data(self, tabdata: TabData):
        saveme = pd.DataFrame(tabdata.scaled)
        saveme.to_csv(self.scaled_data_file, header=False, index=False, sep=str(self.output_delimiter))
        self.computation_done = True
        self.send_data = False
        return True
    
    def save_pca(self, means, std):
        # update PCA and save
        if means is not None:
            pd.DataFrame(means).to_csv(self.means_file, sep=str(self.output_delimiter))
            pd.DataFrame(std).to_csv(self.stds_file, sep=str(self.output_delimiter))
        self.to_csv(self.left_eigenvector_file, self.right_eigenvector_file, self.eigenvalue_file, sep=self.output_delimiter)
    
    def save_explained_variance(self, eigenvalues, variances):
        varex = sh.variance_explained(eigenvalues=eigenvalues, variances=variances)
        pd.DataFrame(varex).to_csv(self.explained_variance_file, sep=str(self.output_delimiter))
    
    def copy_input_to_output(self):
        print('MOVE INPUT TO OUTPUT')
        shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)

    def to_csv(self, H, G, S):
        print('saving pca'+self.left_eigenvector_file)
        print(self.output_delimiter)
        pd.DataFrame(H, index=self.tabdata.rows).to_csv(self.left_eigenvector_file, sep=str(self.output_delimiter))
        pd.DataFrame(G, self.tabdata.columns).to_csv(self.right_eigenvector_file, sep=str(self.output_delimiter))
        pd.DataFrame(S).to_csv(self.eigenvalue_file, sep=str(self.output_delimiter), header=False, index=False)

    def save_projections(self, projections):
        save = pd.DataFrame(projections, index=self.tabdata.columns)
        save.to_csv(self.projection_file, sep=str(self.output_delimiter), header=True, index=True)