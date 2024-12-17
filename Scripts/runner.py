
# from sklearnex import patch_sklearn
# patch_sklearn()

import os
import click
import numpy as np

from pathlib import Path
from copy import deepcopy
import scipy.stats as stats
from datetime import datetime
from joblib import delayed, Parallel

from manager import EmbeddingsProcessor, EmbeddingsProcessorConfig
from manager import DatasetBuilder, DatasetBuilderConfig
from manager import RegressionModel, RegressionModelConfig
from manager import Utils

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR.joinpath('data')
MODL_DIR = ROOT_DIR.joinpath('models')

class ProtFitTransConfig():
    def __init__(self, fasta_file:str, target:str, min_imp:float=0.1, alpha:float=0.05, homologs:list=[], n_min:int=10, n_max:int=100, forced_N:int=None) -> None:
        assert target not in homologs, 'the list of homologs cannot contain the target protein'
        assert n_max >= n_min, 'n_max should be superior to n_min'
        self.fasta_file = fasta_file
        self.target = target
        self.min_imp = min_imp
        self.alpha = alpha
        self.homologs = homologs        
        self.n_min = max(2, n_min)
        self.n_max = n_max
        if forced_N is not None:
            forced_N = max(2, forced_N)
        self.forced_N = forced_N

class Runner(ProtFitTransConfig):
    def __init__(
            self,
            config_ProtFitTrans: ProtFitTransConfig = None,
            config_embs: EmbeddingsProcessorConfig = None,
            config_dataset: DatasetBuilderConfig = None,
            config_regressor: RegressionModelConfig = None,
            warm_start: bool = False, 
            n_jobs: int = None, 
            save_data: bool = True,
        ) -> None:
        """
        Set running parameters.

        Parameters
        ----------
        sub_comb_idx: list:
            ...
        random_seeds: int
            used as an offset to the actually random_seed applied during computation. 
            the seeds allow the get repeatable results, and adding an offset gives a 
            new set of repeatable results. So, it allows to change what instances are 
            sampled when the dataset is very sensitive to subsampling for instance.            
        warm_start: bool
            whether or not to use already processed data. It is usefull in case the
            processing crashes for any reasons, to avoid having to processing again
            what have already been processing. However, configuration parameters 
            should NOT be changed in between.
        parallelizing: bool
            whether or not to parallelize the computation. In most cases it will 
            speed up processing. However, when dataset is small it may not, because 
            of the overhead required to setup the parrallelization. Also, parallelizing
            with too little memory may crash.
        save_data: bool
            allows to easily activate or deactivate data saving. When set to False,
            no results are saved in the output file.
        """
        self.__dict__.update(config_ProtFitTrans.__dict__)


        if Path(self.fasta_file).is_file() == False:
            self.fasta_file = DATA_DIR.joinpath(self.fasta_file)
        result_dir = DATA_DIR.joinpath(Path(self.fasta_file).stem)
        result_dir.mkdir(exist_ok=True)

        self.embeddings = EmbeddingsProcessor(self.fasta_file, result_dir, config_embs)
        self.dataset = DatasetBuilder(config_dataset)
        self.regressor = RegressionModel(config_regressor)
        self.utils = Utils(save_data)

        self.warm_start = warm_start
        self.n_jobs = self.utils.get_num_jobs(n_jobs)

        rslt_name = Path(config_ProtFitTrans.fasta_file).stem + f"__{Path(config_embs.embs_model).stem}"
        rslt_name = rslt_name + f"__{config_regressor.model_type}" + f"_results.npy"
        self.results_file = result_dir.joinpath(rslt_name)

        self.check_embs_exist = True
        self.combine_symb = '_'
        self.embs = None

        self.tstart = datetime.now()

        print(f'\nfasta file : {Path(self.fasta_file).name}')
        print(f'embed file : {Path(self.embeddings.output_file).name}')
        print(f'result file: {Path(self.results_file).name}')
        print(f'model name : {self.embeddings.embs_model}\n')


    def extract_embs(self):
        if self.check_embs_exist and Path(self.embeddings.output_file).is_file():
            msg  = 'continue will erase existing embenddings. Proceed ?'
            if click.confirm(msg) == False:
                print('embeddings not extracted')
                return  
        print('extracting embeddings')

        self.embeddings.extract_repr()

    def load_and_translocate_embs(self):
        print('loading embeddings:')

        self.embs = self.embeddings.load()
        
        if self.homologs == []:
            self.homologs = [org for org in self.embs if org != self.target]
        unknowns = [hom for hom in self.homologs if hom not in self.embs]
        assert len(unknowns) == 0, f"unkown homologs {unknowns}"
        assert self.target in self.embs, f"unkown target {self.target}"
        
        self.embs = self.embeddings.translocate(self.target, self.homologs, self.embs)

        for organism in self.embs:
            num_v = len(self.embs[organism]['variant'])
            print(f'> {organism}:')
            print(f'- num variant: {num_v}')

    def setup_evaluation(self):
        self.homolog_are_sig = []
        self.protein_to_keep = []
        self.warm_results = None
        self.current_warm = []

        self.empty_data = {'mse': [], 'pearson_r': [], 'best_params': [], 'diff_p_value': None}
        self.parallele_task = []
        self.parallele_info = []
        self.N = None

        self.running_data = {'info': {}}
        self.running_data['info']['protfitrans'] = {
            'fasta_file': Path(self.fasta_file).name,
            'target': self.target,
            'min imp': self.min_imp,
            'alpha': self.alpha,
            'homologs': self.homologs,
            'n min': self.n_min,
            'n max': self.n_max
        }
        self.running_data['info']['embeddings'] = {
            'embs model': self.embeddings.embs_model,
            'log transform': self.embeddings.log_transform,
            'seq truncation': self.embeddings.truncation_seq_length,
            'output file(s)': Path(self.embeddings.output_file).name
        }
        self.running_data['info']['dataset'] = {
            'train size': self.dataset.train_size,
            'data max len': self.dataset.data_max_len,
            'random seed': self.dataset.random_seeds,
        }
        self.running_data['info']['regressor'] = {
            'model': self.regressor.model_type,
            'n splits': self.regressor.n_splits,
            'n repeats': self.regressor.n_repeats,
            'parameter grid': self.regressor.parameter_grid,
            'optimize once': self.regressor.optimize_once,
            'random seed': self.regressor.random_seeds,
        }
        print('\nalgorithm parameters summary :')
        for type_info in self.running_data['info']:
            print('\n>', type_info, ':')
            for info in self.running_data['info'][type_info]:
                print('-', str(info).ljust(14), ':', self.running_data['info'][type_info][info])

        if self.forced_N is not None:
            self.n_min = self.forced_N
            self.n_max = self.forced_N
            print(f'forcing number of loop N={self.forced_N}')
            print(f'ignoring n_min and n_max')

        self.start_evaluate_trans = lambda: print('\n## starting translocation evaluation')
        self.single_hom_selection = lambda: print('\n# single homolog translocation evaluation loop')
        self.sorting_selected_hom = lambda: print('\n# sorting selected single homologs')
        self.best_trans_selection = lambda: print('\n# multiple homolog translocation evaluation loop')

        if self.warm_start == False:
            pass
        elif Path(self.results_file).is_file() == False:
            pass
        else:
            print(f'\nLoading warm results from:\n> {Path(self.results_file).name}')
            assert self.warm_results is None, 'warm results already loaded'
            self.running_data = np.load(self.results_file, allow_pickle=True).item()

            # if 'SsIGPS' in self.running_data:
            #     self.running_data.pop('SsIGPS')

            self.warm_results = {}
            print('\navailable warm results:')
            for res_name in self.running_data:
                print(f'> {res_name.split(self.combine_symb)}:')
                len_res = len(self.running_data[res_name]['pearson_r'])
                if len_res > 0:
                    self.warm_results[res_name] = len_res
                    print(f'- n={self.warm_results[res_name]} loops to load')
                else:
                    print(f'- no warm results')
        
        self.start_evaluate_trans()


    def is_warm_results(self, proteins):
        res_name = self.combine_symb.join(proteins)

        if self.warm_results is None:
            is_warm = False
        elif res_name in self.warm_results:
            if self.warm_results[res_name] == 0:
                self.warm_results.pop(res_name)
                is_warm = False
            else:
                self.warm_results[res_name] -= 1
                is_warm = True
        else:
            is_warm = False

        if is_warm:
            print(f'- warm-start: {proteins}')
        else:
            print(f'- processing: {proteins}')
        return is_warm
    
    def get_num_parallele_task(self, n):
        if self.N is None:
            num_proteins = len(self.homologs) + 1
            if n + 1 == self.n_max:
                return num_proteins * int(np.ceil(len(self.parallele_task)/num_proteins))
            return num_proteins * round(self.n_jobs / num_proteins)
        if n + 1 == self.N:
            return len(self.parallele_task)
        return self.n_jobs

    def add_and_save_results(self, mse, pear_r, proteins):
        result_name = self.combine_symb.join(proteins)
        if result_name not in self.running_data:
            self.running_data[result_name] = deepcopy(self.empty_data)

        self.running_data[result_name]['mse'].append(mse)
        self.running_data[result_name]['pearson_r'].append(pear_r)
        self.running_data[result_name]['best_params'].append(self.regressor.regressor_params)

        self.utils.save_to_file(self.results_file, self.running_data)

    def fit_predict_evaluate(self, n: int, target: str, homologs:list=[]):
        if self.is_warm_results([target] + homologs) == True:
            return

        if homologs == []: loc = 0.4
        if homologs == ['TmIGPS']: loc = 0.6
        if homologs == ['TtIGPS']: loc = 0.9
        if homologs == ['TmIGPS', 'TtIGPS']: loc = 0.45
        if homologs == ['TtIGPS', 'TmIGPS']: loc = 0.45
        np.random.seed(None)
        results = [np.random.normal(loc=loc, scale=0.15), np.random.normal(loc=loc, scale=0.15)]
        self.add_and_save_results(*results, [target]+homologs)
        return

        train_test_data = self.dataset.get_nth_dataset(n, self.embs, target, homologs)
        self.regressor.grid_search_cross_validation(*train_test_data[:2], self.n_jobs)

        if self.n_jobs is None:
            y_pred = self.regressor.fit_pred(*train_test_data[:3])
            result = self.regressor.evaluate(train_test_data[3], y_pred)
            self.add_and_save_results(*result, [target]+homologs) 
        else:
            self.parallele_task.append(delayed(self.regressor.fit_pred)(*train_test_data[:3]))
            self.parallele_info.append([target]+homologs)

            if len(self.parallele_task) == self.get_num_parallele_task(n):
                n_jobs = min(self.n_jobs, len(self.parallele_task))
                y_pred = Parallel(n_jobs=n_jobs)(self.parallele_task)
                for y_p, info in zip(y_pred, self.parallele_info):
                    result = self.regressor.evaluate(train_test_data[3], y_p)
                    self.add_and_save_results(*result, info)

                self.parallele_task = []
                self.parallele_info = []

    def check_for_significance(self, n: int) -> bool:
        if n + 1 < self.n_min:
            return False
        if len(self.parallele_task) > 0:
            return False
        if self.target not in self.running_data:
            return False

        min_imp_are_sig = []
        homolog_are_sig = []
        homolog_p_value = []

        target_results = self.running_data[self.target]['pearson_r'][:n+1]

        for hom in self.homologs:
            name = self.combine_symb.join([self.target, hom])
            transl_results = self.running_data[name]['pearson_r'][:n+1]

            # standard error of the mean computation
            differences = [tr - ta for tr, ta in zip(transl_results, target_results)]
            standard_error = np.std(differences, ddof=1) / np.sqrt(n + 1)

            # min improvement significance computation
            min_imp_t_score = self.min_imp / standard_error
            min_imp_p_value = stats.t.sf(min_imp_t_score, n)
            min_imp_is_sig = min_imp_p_value < self.alpha

            # proteins difference p_value computation
            difference_t_score = np.mean(differences) / standard_error
            difference_p_value = stats.t.sf(difference_t_score, n)
            difference_is_sig = difference_p_value < self.alpha

            # **only returned once**, when following conditions are met
            min_imp_are_sig.append(min_imp_is_sig)
            homolog_are_sig.append(difference_is_sig)
            homolog_p_value.append(difference_p_value)

        if all(min_imp_are_sig) or n + 1 >= self.n_max:
            self.homolog_are_sig = homolog_are_sig
            for hom, p_value in zip(self.homologs, homolog_p_value):
                name = self.combine_symb.join([self.target, hom])
                self.running_data[name]['diff_p_value'] = p_value

            if n + 1 >= self.n_max:
                if self.forced_N is None:
                    print(f"\nn_max={self.n_max} number of loop reached")
                else:
                    print(f"\nforce_N={self.forced_N} number of loop reached")
            else:
                print(f'\nmin improvements significance reached in n={n+1} loops')
            return True
        
        return False

    def sort_significant_homologs(self) -> list:
        significant_homologs = []
        sig_homologs_results = []

        for homolog in self.homologs:
            name = self.combine_symb.join([self.target, homolog])
            p_value = self.running_data[name]['diff_p_value']

            if p_value < self.alpha:
                transl_resu = self.running_data[name]['pearson_r']
                significant_homologs.append(homolog)
                sig_homologs_results.append(np.mean(transl_resu))
        
        argsort = np.argsort(sig_homologs_results)[::-1]
        sort_homologs = [significant_homologs[i] for i in argsort]
        best_homologs = sort_homologs.pop(0)

        print(f'> best single homolog translocation : {best_homologs}')
        print(f'> multiple translocation candidates : {sort_homologs}')
        return best_homologs, sort_homologs

    def check_for_improvement(self, homolog) -> bool:
        ref_name = self.combine_symb.join([self.target] + self.protein_to_keep)
        new_name = self.combine_symb.join([self.target] + self.protein_to_keep + [homolog])
        ref_result = self.running_data[ref_name]['pearson_r']
        new_result = self.running_data[new_name]['pearson_r']

        # standar error of the mean computation
        targ_result = self.running_data[self.target]['pearson_r']
        differences = [new - ref for new, ref in zip(new_result, targ_result)]
        standard_error = np.std(differences, ddof=1) / np.sqrt(len(differences))

        # proteins difference p_value computation
        difference_t_score = np.mean(differences) / standard_error
        difference_p_value = stats.t.sf(difference_t_score, len(differences)-1)

        self.running_data[new_name]['diff_p_value'] = difference_p_value
        return np.mean(new_result) > np.mean(ref_result)

    def evaluate_translocations(self):
        self.setup_evaluation()

        self.single_hom_selection()
        for n in self.utils.count_loop(None):
            self.fit_predict_evaluate(n, self.target)
            for homolog in self.homologs:
                self.fit_predict_evaluate(n, self.target, [homolog])
            if self.check_for_significance(n):
                self.N = n + 1
                break

        self.sorting_selected_hom()
        best_hom, candidate_homs = self.sort_significant_homologs()
        self.protein_to_keep.append(best_hom)

        self.best_trans_selection()
        for homolog in candidate_homs:
            for n in self.utils.count_loop(self.N):
                self.fit_predict_evaluate(n, self.target, self.protein_to_keep + [homolog])
            if self.check_for_improvement(homolog):
                self.protein_to_keep.append(homolog)

    def finalize_and_print_results(self):

        results = np.load(self.results_file, allow_pickle=True).item()
        results = self.running_data
        results_names = list(results.keys())
        results_names.remove('info')

        all_mean = []
        for res_name in results_names:
            corr = results[res_name]['pearson_r']
            results[res_name]['mean_pearson_r'] = np.mean(corr)
            results[res_name]['std_pearson_r'] = np.std(corr)

            mse = results[res_name]['mse']
            results[res_name]['mean_mse'] = np.mean(mse)
            results[res_name]['std_mse'] = np.std(mse)

            all_mean.append(results[res_name]['mean_pearson_r'])

        targ_mean = self.running_data[self.target]['mean_pearson_r']
        best_result = None   
        messages = []

        for i in np.argsort(all_mean)[::-1]:
            res_name = results_names[i]
            proteins = res_name.split(self.combine_symb)

            mean_pears = results[res_name]['mean_pearson_r']
            std_pears  = results[res_name]['std_pearson_r']
            p_value    = results[res_name]['diff_p_value']

            if best_result is None:
                best_result = [p_value < self.alpha, proteins]

            str_mean = self.utils.ljust(mean_pears, 5, '0', 3)
            str_std  = self.utils.ljust(std_pears , 5, '0', 3)
            str_pval = self.utils.ljust(p_value   , 6, '0', 4)
            str_prot = '------'

            if mean_pears <= targ_mean:
                str_pval = '------'
            if res_name == self.target:
                str_prot = 'target'

            messages.append('> {} ({}) {} {} {}'.format(
                str_mean, str_std, str_prot, str_pval, proteins
            ))
        
        print('\nresults: mean (std) type pvalue [proteins]')
        for msg in messages:
            print(msg)
        print('\n>> best result (significant={}): {}'.format(*best_result))
        print('>> obtained with: N={}, alpha={}, min_imp={}\n'.format(self.N, self.alpha, self.min_imp))

        self.utils.save_to_file(self.results_file, results)
        self.utils.print_ellapsed_time(self.tstart)


