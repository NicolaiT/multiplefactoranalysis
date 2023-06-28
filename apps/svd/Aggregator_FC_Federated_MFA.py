from apps.svd.FC_Federated_MFA import FCFederatedMFA
import  numpy as np
from apps.svd.Steps import Step
import scipy as sc
import scipy.sparse.linalg as lsa
import scipy.linalg as la
from apps.svd.shared_functions import eigenvector_convergence_checker
from apps.svd.algo_params import QR, PCA_TYPE
import apps.svd.shared_functions as sh
from apps.svd.COParams import COParams

class AggregatorFCFederatedMFA(FCFederatedMFA):
    def __init__(self):
        self.dummy = None
        self.coordinator = True
        self.values_per_row = []
        self.number_of_samples = []

        FCFederatedMFA.__init__(self)
        
    def convert_incoming(self, incomings):
        incoming_omics = [[]] * len(incomings[0])
        for idx in range(len(incoming_omics)):
            for incoming in incomings:
                incoming_omics[idx].append(incoming[idx])
        return incoming_omics
        
    def unify_row_names(self, incomings):
        '''
        Make sure the clients use a set of common row names.
        Make sure the maximal fraction of NAs is not exceeded.

        Parameters
        ----------
        incoming Incoming data object from clients

        Returns
        -------

        '''
        incoming_omics = self.convert_incoming(incomings)
        self.outs = []
        print("incoming_omics")
        print(incoming_omics)
        for idx, incoming in enumerate(incoming_omics):
            print(incoming)
            mysample_count = 0
            myintersect = set(incoming[0][COParams.ROW_NAMES.n])

            nandict = {}
            for s in incoming:
                for n, v in zip(s[COParams.ROW_NAMES.n], s[COParams.NAN.n]):
                    if n in nandict:
                        nandict[n] = nandict[n]+v
                    else:
                        nandict[n] = v
                myintersect = myintersect.intersection(set(s[COParams.ROW_NAMES.n]))
                mysample_count = s[COParams.SAMPLE_COUNT.n]+mysample_count

            select = []
            for n in nandict:
                fract = nandict[n]/mysample_count
                if fract<=self.max_nan_fraction:
                    select.append(n)

            print(select)
            myintersect = myintersect.intersection(set(select))
            self.total_sampels.append(mysample_count)
            print("idx", idx)
            print("incomings len", len(incomings))
            print("self.k len", len(self.k))
            self.outs[idx] = {COParams.PCS.n: self.k[idx], COParams.SEND_PROJ: self.send_projections}
            newrownames = list(myintersect)
            self.outs[idx][COParams.ROW_NAMES.n] = newrownames

            values_per_row = []
            for n in newrownames:
                values_per_row.append(mysample_count-nandict[n])
            self.values_per_row.append(values_per_row)
            print(newrownames)
            print(values_per_row)
            print('[API] [COORDINATOR] row names identified!')
    
    def compute_means(self, incomings):
        incoming_omics = self.convert_incoming(incomings)
        self.outs = []
        for idx, incoming in enumerate(incoming_omics):
            print(incoming)
            my_sums = []
            my_samples = 0

            for s in incoming:
                my_sums.append(s[COParams.SUMS.n])
                my_samples = my_samples+s[COParams.SAMPLE_COUNT.n]

            my_sums = np.stack(my_sums)
            my_sums = np.nansum(my_sums, axis=0)

            my_sums = my_sums/self.values_per_row[idx]
            print('SUMS')
            print(my_sums)

            self.outs[idx] = {COParams.MEANS.n : my_sums }
            self.number_of_samples.append(my_samples)
    
    def compute_std(self, incomings):
        incoming_omics = self.convert_incoming(incomings)
        self.outs = []
        for idx, incoming in enumerate(incoming_omics):
            my_ssq  = []
            for s in incoming:
                print(s[COParams.SUM_OF_SQUARES.n])
                my_ssq.append(s[COParams.SUM_OF_SQUARES.n])
            my_ssq = np.stack(my_ssq)
            my_ssq = np.nansum(my_ssq, axis=0)
            print('COMPUTE STD')
            print(my_ssq)
            val_per_row = [v-1 for v in self.values_per_row[idx]]
            variances = my_ssq/(val_per_row)
            my_ssq = np.sqrt(variances)
            self.std.append(my_ssq)
            print('STD')
            print(self.std[idx])

            if self.perc_highly_var is not None:
                hv = int(np.floor(self.tabdatas[idx].scaled.shape[0] * self.perc_highly_var))
            else:
                # select all genes
                hv = self.tabdatas[idx].scaled.shape[0]


            remove = np.where(self.std[idx].flatten()==0)
            # std in fact contains the standard deviation
            select = np.argsort(self.std[idx].flatten())[0:hv]

            REM = self.tabdatas[idx].rows[remove]
            SEL = self.tabdatas[idx].rows[select]
            print(select)
            print(SEL)
            print(remove)
            print(REM)
            self.outs[idx] = {COParams.STDS.n : self.std, COParams.SELECT.n: select, COParams.REMOVE.n: remove, COParams.VARIANCES.n: variances}