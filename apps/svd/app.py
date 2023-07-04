import time

from engine.app import AppState, app_state, Role

from apps.svd.config import FCConfig
from apps.svd.Aggregator_FC_Federated_PCA import AggregatorFCFederatedPCA
from apps.svd.Client_FC_FederatedPCA import ClientFCFederatedPCA
from apps.svd.FC_Federated_MFA import FCFederatedMFA
from apps.svd.Steps import Step
from apps.svd.COParams import COParams
import copy
import numpy as np




# This is the first (initial) state all app instances are in at the beginning
# By calling it 'initial' the FeatureCloud template engine knows that this state is the first one to go into automatically at the beginning


@app_state('initial')  # The first argument is the name of the state ('initial'), the second specifies which roles are allowed to have this state (here BOTH)
class InitialState(AppState):

    def configure(self):
        print("[CLIENT] Parsing parameter file...", flush=True)
        if self.id is not None:  # Test is setup has happened already
            # parse parameters
            self.config = FCConfig()
            self.config.parse_configuration()
            self.store('configuration', self.config)
            print("[CLIENT] finished parsing parameter file.", flush=True)

    def register(self):
        self.register_transition('init_pca', Role.BOTH)
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        self.store('idx_omics', 0)
        self.store('data', [])
        self.store('eigen_values', [])
        self.store('global_pca', False)
        self.configure()
        print('[STARTUP] Instantiate SVD')
        self.progress_increment = 1/(20+self.config.k*2)
        
        if not self.config.config_available:
            return 'terminal'
        else:
            return 'init_pca'
        
        

@app_state('init_pca', Role.BOTH)
class InitPCA(AppState):
    def register(self):
        self.register_transition('check_row_names', Role.COORDINATOR)
        self.register_transition('wait_for_params', Role.PARTICIPANT)
        self.register_transition('terminal', Role.BOTH)
    
    def run(self):
        if self.is_coordinator:
            self.store('svd' ,AggregatorFCFederatedPCA())
        else:
            self.store('svd', ClientFCFederatedPCA())
        self.load('svd').step = Step.LOAD_CONFIG
        self.load('svd').copy_configuration(self.load('configuration'))
        
        if self.load('global_pca'):
            self.load('svd').set_tabdata(self.load('global_pca_data')) 
            self.load('svd').center = False
            self.load('svd').L2_norm = False         
        else:
            input_files = self.load('svd').input_files
            number_of_omics = len(input_files)
        
            if number_of_omics == 0:
                raise KeyError('No data found. Please ensure that the data file is present and correctly named.')
            self.store('n_omics', number_of_omics)
            # if not isinstance(self.load('idx_omics'), int):
            
            
            self.load('svd').set_input_file(input_files[self.load('idx_omics')])
            print('[STARTUP] Configuration copied')
            
            # READ INPUT DATA
            self.load('svd').read_input_files()
            
        out = self.load('svd').out
        print("out", out)
        self.send_data_to_coordinator(out)
            
        if self.is_coordinator:
            return 'check_row_names'
        else:
            return 'wait_for_params'
            


@app_state('check_row_names', Role.COORDINATOR)
class CheckRowNames(AppState):
    '''
    This state collects all relevant parameters necessary to unify the run.
    Notably it makes sure the names of the variables match.

    '''
    def register(self):
        self.register_transition('wait_for_params', Role.COORDINATOR)


    def run(self):
        print('gathering')
        incoming = self.gather_data()
        print('unifying row names') 
        self.load('svd').unify_row_names(incoming) # CRASH 'NoneType' object is not subscriptable
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'wait_for_params'



@app_state('wait_for_params', Role.BOTH)
class WaitForParamsState(AppState):
    '''
    This state collects all relevant parameters necessary to unify the run.
    Notably it makes sure the names of the variables match.

    '''
    def register(self):
        self.register_transition('aggregate_sums', Role.COORDINATOR)
        self.register_transition('compute_std', Role.PARTICIPANT)
        self.register_transition('start_power_iteration', Role.BOTH)

    def run(self):
        incoming = self.await_data(is_json=False)
        print('setting parameters')
        self.load('svd').set_parameters(incoming)
        self.load('svd').select_rows(incoming)
        config = self.load('configuration')
        #if not config.center:
        #    return 'start_power_iteration'



        self.load('svd').compute_sums()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)
        if self.is_coordinator:
            return 'aggregate_sums'
        else:
            return 'compute_std'


@app_state('aggregate_sums', Role.COORDINATOR)
class AggregateSummaryStatsState(AppState):
    def register(self):
        self.register_transition('compute_std', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        print('setting parameters')
        self.load('svd').compute_means(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'compute_std'


@app_state('compute_std', Role.BOTH)
class ComputeSummaryStatsState(AppState):
    def register(self):
        self.register_transition('aggregate_stds', Role.COORDINATOR)
        self.register_transition('apply_scaling', Role.PARTICIPANT)

    def run(self):
        incoming = self.await_data()
        self.load('svd').compute_sum_of_squares(incoming)
        print("GLOBAL SAMPLE COUNT", self.load('svd').global_sample_count)
        out = self.load('svd').out
        self.send_data_to_coordinator(out)
        if self.is_coordinator:
            return 'aggregate_stds'
        else:
            return 'apply_scaling'

@app_state('aggregate_stds', Role.COORDINATOR)
class AggregateStdState(AppState):
    def register(self):
        self.register_transition('apply_scaling', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        print('setting parameters')
        self.load('svd').compute_std(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'apply_scaling'

@app_state('start_power_iteration', Role.BOTH)
class PowerIterationStartState(AppState):
    def register(self):
        self.register_transition('aggregate_h', Role.COORDINATOR)
        self.register_transition('update_h', Role.PARTICIPANT)
        self.register_transition('aggregate_randomized_projections_cov', Role.COORDINATOR)
        self.register_transition('compute_final_h_cov', Role.PARTICIPANT)

    def run(self):
        self.store('covariance_only', False)
        number_features = self.load('svd').tabdata.scaled.shape[0]
        if number_features <= 20 * self.load('svd').k:
            self.store('covariance_only', True)
            self.load('svd').compute_covariance()
            self.load('svd').init_federated_qr()

        else:
            self.load('svd').init_random()
            self.load('svd').init_power_iteration()
            self.load('svd').init_federated_qr()

        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.load('covariance_only'):
            if self.is_coordinator:
                return 'aggregate_randomized_projections_cov'
            else:
                return 'compute_final_h_cov'
        else:
            if self.is_coordinator:
                return 'aggregate_h'
            else:
                return 'update_h'


@app_state('apply_scaling', Role.BOTH)
class ScaleDataState(AppState):

    def register(self):
        self.register_transition('mfa_prerequisites', Role.BOTH)


    def run(self):
        config = self.load('configuration')
        incoming = self.await_data()
        self.load('svd').apply_scaling(incoming, highly_variable=config.highly_variable)
        current_data = self.load('svd').tabdata
        print("CURRENT DATA: ", current_data)
        self.store('current_tabdata', current_data)
        return 'mfa_prerequisites'

@app_state('mfa_prerequisites', Role.BOTH)
class MFAPrerequisites(AppState):
    def register(self):
        self.register_transition('start_power_iteration', Role.BOTH)
    
    def run(self):
        print('Calling start_power_iteration')
        return 'start_power_iteration'

@app_state('aggregate_h', Role.COORDINATOR)
class AggregateHState(AppState):
    def register(self):
        self.register_transition('update_h', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_h(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'update_h'

@app_state('aggregate_h_final', Role.COORDINATOR)
class AggregateHState(AppState):
    def register(self):
        self.register_transition('compute_g_and_local_norm', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_h(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'compute_g_and_local_norm'


@app_state('update_h', Role.BOTH)
@app_state('update_h_2', Role.BOTH)
class ComputeGState(AppState):
    def register(self):
        self.register_transition('update_h', Role.PARTICIPANT)
        self.register_transition('aggregate_h', Role.COORDINATOR)
        self.register_transition('compute_final_h', Role.PARTICIPANT)
        self.register_transition('aggregate_randomized_projections', Role.COORDINATOR)

    def run(self):
        incoming = self.await_data()
        next = self.load('svd').update_h(incoming)
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            if next == 'iter':
                return 'aggregate_h'
            else:
                return 'aggregate_randomized_projections'
        else:
            if next == 'iter':
                return 'update_h'
            else:
                return 'compute_final_h'


@app_state('aggregate_randomized_projections', Role.COORDINATOR)
class AggregateRandomizedProjections(AppState):
    def register(self):
        self.register_transition('compute_final_h', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_randomized(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'compute_final_h'

@app_state('aggregate_randomized_projections_cov', Role.COORDINATOR)
class ComputeRandomizedProjections(AppState):
    def register(self):
        self.register_transition('compute_final_h_cov', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_randomized(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'compute_final_h_cov'




@app_state('compute_final_h', Role.BOTH)
class ComputeFinalH(AppState):
    def register(self):
        self.register_transition('compute_local_conorm', Role.PARTICIPANT)
        self.register_transition('aggregate_local_norm', Role.COORDINATOR)

    def run(self):
        incoming = self.await_data()
        self.load('svd').project_u(incoming)
        self.load('svd').compute_local_eigenvector_norm()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_local_norm'
        else:
            return 'compute_local_conorm'

@app_state('compute_final_h_cov', Role.BOTH)
class ComputeFinalHCov(AppState):
    def register(self):
        self.register_transition('compute_local_conorm', Role.PARTICIPANT)
        self.register_transition('aggregate_local_norm', Role.COORDINATOR)

    def run(self):
        incoming = self.await_data()
        self.load('svd').project_u_cov(incoming)
        self.load('svd').compute_local_eigenvector_norm()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_local_norm'
        else:
            return 'compute_local_conorm'

@app_state('compute_g_and_local_norm', Role.BOTH)
class ComputeLocalGandNormState(AppState):
    def register(self):
        self.register_transition('compute_local_conorm', Role.PARTICIPANT)
        self.register_transition('aggregate_local_norm', Role.COORDINATOR)

    def run(self):
        incoming = self.await_data()
        self.load('svd').compute_g(incoming)
        self.load('svd').compute_local_eigenvector_norm()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)
        if self.is_coordinator:
            return 'aggregate_local_norm'
        else:
            return 'compute_local_conorm'

@app_state('compute_local_norm', Role.BOTH)
class ComputeLocalNormState(AppState):
    def register(self):
        self.register_transition('compute_local_conorm', Role.PARTICIPANT)
        self.register_transition('aggregate_local_norm', Role.COORDINATOR)
        self.register_transition('normalize_g', Role.PARTICIPANT)
        self.register_transition('normalize_g_update_h', Role.PARTICIPANT)


    def run(self):
        incoming = self.await_data()
        self.load('svd').orthogonalise_current(incoming)
        done = self.load('svd').compute_local_eigenvector_norm()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_local_norm'
        else:
            if done:
                if self.load('covariance_only'):
                    return 'normalize_g'
                else:
                    return 'normalize_g_update_h'
            else:
                return 'compute_local_conorm'


@app_state('aggregate_local_norm', Role.COORDINATOR)
class AggregateLocalNormState(AppState):

    def register(self):
        self.register_transition('compute_local_conorm', Role.COORDINATOR)
        self.register_transition('normalize_g', Role.COORDINATOR)
        self.register_transition('normalize_g_update_h', Role.COORDINATOR)


    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_eigenvector_norms(incoming)
        done = self.load('svd').orthonormalisation_done
        out = self.load('svd').out
        self.broadcast_data(out)
        if done:
            if self.load('covariance_only'):
                return 'normalize_g'
            else:
                return 'normalize_g_update_h'
        else:
            return 'compute_local_conorm'



@app_state('compute_local_conorm', Role.BOTH)
class ComputeLocalConormState(AppState):
    def register(self):
        self.register_transition('compute_local_norm', Role.PARTICIPANT)
        self.register_transition('aggregate_local_conorm', Role.COORDINATOR)
        self.register_transition('normalize_g', Role.BOTH)


    def run(self):
        incoming = self.await_data()
        self.load('svd').calculate_local_vector_conorms(incoming)
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_local_conorm'
        else:
            return 'compute_local_norm'

@app_state('aggregate_local_conorm', Role.COORDINATOR)
class AggregateLocalCoNormState(AppState):
    def register(self):
        self.register_transition('compute_local_norm', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_conorms(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'compute_local_norm'

@app_state('normalize_g_update_h', Role.BOTH)
class NormalizeGState(AppState):
    def register(self):
        self.register_transition('get_h_and_finish', Role.PARTICIPANT)
        self.register_transition('aggregate_h_and_finish', Role.COORDINATOR)

    def run(self):
        incoming = self.await_data()
        self.load('svd').normalise_orthogonalised_matrix(incoming)
        self.load('svd').send_h()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_h_and_finish'
        else:
            return 'get_h_and_finish'


@app_state('aggregate_h_and_finish',Role.COORDINATOR)
class AggregateHAndFinishState(AppState):
    def register(self):
        self.register_transition('get_h_and_finish', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('svd').aggregate_h(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        return 'get_h_and_finish'

@app_state('get_h_and_finish', Role.BOTH)
class NormalizeGState(AppState):
    def register(self):
        self.register_transition('separate_pca', Role.BOTH) # save_results
        self.register_transition('share_projections', Role.COORDINATOR)

    def run(self):
        config = self.load('configuration')
        incoming = self.await_data()
        self.load('svd').save_h(incoming)
        self.load('svd').compute_projections()

        if config.send_projections:
            out = self.load('svd').out
            self.send_data_to_coordinator(out)

        if config.send_projections and self.is_coordinator:
            return 'share_projections'
        else:
            return 'separate_pca' # save_results



@app_state('normalize_g', Role.BOTH)
class NormalizeGState(AppState):
    def register(self):
        self.register_transition('separate_pca', Role.BOTH) # save_results
        self.register_transition('share_projections', Role.COORDINATOR)

    def run(self):
        config = self.load('configuration')
        incoming = self.await_data()
        self.load('svd').normalise_orthogonalised_matrix(incoming)
        self.load('svd').compute_projections()

        if config.send_projections:
            out = self.load('svd').out
            self.send_data_to_coordinator(out)

        if config.send_projections and self.is_coordinator:
            return 'share_projections'
        else:
            return 'separate_pca' # save_results


@app_state('share_projections', Role.COORDINATOR)
class ShareProjectionsState(AppState):
    def register(self):
        self.register_transition('separate_pca', Role.COORDINATOR) 

    def run(self):
        incoming = self.gather_data()
        self.load('svd').redistribute_projections(incoming)
        out = self.load('svd').out
        self.broadcast_data(out)
        print('Starting separate pca')
        return 'separate_pca' 

@app_state('separate_pca', Role.BOTH)
class SeparatePCA(AppState):
    def register(self):
        self.register_transition('init_pca', Role.BOTH)
        self.register_transition('scale_tabdata', Role.BOTH)
        self.register_transition('projection_matrix', Role.BOTH)
        
    def run(self) -> str:
        eigen_values = self.load('svd').pca.S
        print("singular values", eigen_values)
        tabdata = self.load('current_tabdata')
        print('data', tabdata.scaled)
        
        new_data = self.load('data')
        new_data.append(tabdata)
        new_eigen_values = self.load('eigen_values')
        new_eigen_values.append(eigen_values)
        
        self.store('data', new_data)
        self.store('eigen_values', new_eigen_values)
        
        if self.load('idx_omics')+1 != self.load('n_omics'):
            new_idx_omics = self.load('idx_omics') + 1
            self.store('idx_omics', new_idx_omics)
            
            return 'init_pca'
        elif self.load('global_pca'):
            print('Starting factor analysis')
            return 'projection_matrix'
        else:
            print('Starting scale tabdata')
            return 'scale_tabdata'
    
@app_state('scale_tabdata', Role.BOTH)
class ScaleTabdata(AppState):
    def register(self):
        self.register_transition('merging_tabdata', Role.BOTH)
        
    def run(self):
        data = self.load('data') # 2 x tabdata
        print('data: ', data)
        eigen_values = self.load('eigen_values') # 2 x 17
        first_singular_values = [eigenvalue[0] for eigenvalue in eigen_values] # 1 x 2 #singular values
        for idx, omic in enumerate(data):
            omic.scaled = omic.scaled / first_singular_values[idx]   
        print('Starting merge tabdata')
        return 'merging_tabdata'

@app_state('merging_tabdata', Role.BOTH)
class MergingTabdata(AppState):
    def register(self):
        self.register_transition('init_pca', Role.BOTH)
        
    def run(self):
        data = self.load('data')
        merged_tabdata = self.merge(data)
        print('merged_data', merged_tabdata.scaled)
        self.store('global_pca_data', merged_tabdata)
        self.store('global_pca', True)
        print('Starting global pca')
        return 'init_pca'
        
    def merge(self, tabData_list):
        merged_tb = copy.deepcopy(tabData_list[0])
        for tabdata in tabData_list[1:]:
            merged_tb.data = np.concatenate((merged_tb.data, tabdata.data),axis=0)
            merged_tb.rows = np.concatenate((merged_tb.rows,tabdata.rows))
            merged_tb.scaled = np.concatenate((merged_tb.scaled,tabdata.scaled),axis=0)

        return merged_tb
    
@app_state('projection_matrix', Role.BOTH)
class ProjectionMatrix(AppState):
    def register(self):
        self.register_transition('factor_scores', Role.BOTH)
    
    def run(self):
        self.generate_M()
        self.projection_matrix()
        return 'factor_scores'
    
    def projection_matrix(self): # implement
        M = self.load('M')
        U = self.load('svd').pca.G
        S = self.load('svd').pca.S
        S = np.diag(S)
        print("M", M)
        print("M.shape", M.shape)
        print("U", U)
        print("U.shape", U.shape)
        print("S", S)
        print("S.shape", S.shape)
        P = np.dot(np.linalg.inv(np.sqrt(M)), np.dot(U, np.linalg.inv(S)))
        print("P", P)
        print("P.shape", P.shape)
        self.store('P', P)
    
    def generate_M(self):
        U = self.load('svd').pca.G
        n = len(U)
        M = np.zeros((n,n))
        np.fill_diagonal(M, 1/self.load('svd').global_sample_count)
        self.store('M', M)
    
@app_state('factor_scores', Role.BOTH)
class FactorScores(AppState):
    def register(self):
        self.register_transition('global_factor_score', Role.BOTH)

    def run(self):
        self.F_omics() #uncomment
        return 'global_factor_score'
        
    def F_omics(self):
        T = self.load('n_omics')
        P = self.load('P')
        print("n_omics: ", T)
        print("P:", P)
        data = self.load('data')
        F_omics = []
        for i in range(T):
            Z = data[i].scaled
            print("Z", Z)
            print("Z.shape", Z.shape)
            F = T * np.dot((np.dot(np.transpose(Z), Z)), P)
            F_omics.append(F)
        self.store('F_omics', F_omics)
    
@app_state('global_factor_score', Role.BOTH)
class GlobalFactorScore(AppState):
    def register(self):
        self.register_transition('inertia', Role.BOTH)
    
    def run(self):
        self.F()
        return 'inertia'
    
    def F(self):
        M = self.load('M')
        U = self.load('svd').pca.G
        S = self.load('svd').pca.S
        S = np.diag(S)
        print("M", M)
        print("M.shape", M.shape)
        print("U", U)
        print("U.shape", U.shape)
        print("S", S)
        print("S.shape", S.shape)
        F = np.dot(np.linalg.inv(np.sqrt(M)), np.dot(U, S))
        print("F", F)
        print("F.shape", F.shape)
        self.store('F', F)
    
@app_state('inertia', Role.BOTH)
class Intertia(AppState):
    def register(self):
        self.register_transition('mfa_final', Role.BOTH)
        
    def run(self):
        self.inertia()
        return 'mfa_final'
    
    def inertia(self):
        S = self.load('svd').pca.S
        # Converting singular values to eigenvalues
        eigenvalues = [s**2 for s in S]
        total = np.sum(eigenvalues)
        inertia = []
        for eigen_value in eigenvalues:
            inertia.append(eigen_value/total)
        self.store('inertia', inertia)

@app_state('mfa_final', Role.BOTH)
class MFAFinal(AppState):
    def register(self):
        self.register_transition('save_results')

    def run(self):
        print('Starting save results')
        return 'save_results'

@app_state('save_results', Role.BOTH)
class ShareProjectionsState(AppState):
    def register(self):
        self.register_transition('finalize', Role.BOTH)

    def run(self):
        print('SAVING RESULTS')
        config = self.load('configuration')
        if config.send_projections:
            # Only wait for projections if they are actually send.
            incoming = self.await_data()
            self.load('svd').save_projections(incoming)
        else:
            # save only local projections
            self.load('svd').save_projections()
        inertia = self.load('inertia')
        F = self.load('F')
        F_omics = self.load('F_omics')
        P = self.load('P')
        self.load('svd').save_MFA(inertia, F, F_omics, P)
        print("inertia: ", inertia)
        self.load('svd').save_explained_variance()
        self.load('svd').save_pca()
        self.load('svd').save_scaled_data()
        self.load('svd').save_logs()
        self.load('svd').copy_input_to_output()
        out = self.load('svd').out
        self.send_data_to_coordinator(out)
        return 'finalize'


@app_state('finalize')
class FinalizeState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        # Wait until all send the finished flag
        if self.is_coordinator:
            self.gather_data()
        return 'terminal'



