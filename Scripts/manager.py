
import os
import sys
import esm
import torch
import shutil
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from esm import FastaBatchedDataset, pretrained, MSATransformer

class EmbeddingsProcessorConfig():
    def __init__(self, embs_model:str='esm2_t33_650M_UR50D', extract:bool=True, log_transform:bool=False, max_num_embs_per_file:int=8000,
                 toks_per_batch:int=4066, truncation_seq_length:int=1022, nogpu:bool=False, warm_start:bool=True) -> None:
        """
        Configuration of the embeddings extraction, that uses fair-esm extractor (https://github.com/facebookresearch/esm)

        Parameters
        ----------
        embs_model: str.
            name of the ESM model used to extract the embeddings. It is automatically downloaded.
            All models name are availble at https://github.com/facebookresearch/esm
        extract: bool.
            to easily activate or deactivate extraction.
        log_transform: bool.
            Whether or not to log transform the fitness values.
        max_num_embs_per_file: int.
            Maximum sizes of the final files containing the embeddings. The embeddings are first 
            extracted in small batches and then gathered in bigger files.
        toks_per_batch: int.
            number of token used per batch, and therefore the number of sequence. the latter depends
            on the number of token (i.e amino acids) contained per sequence.
        truncation_seq_length: int.
            limit the lenght of the input sequences. the ESM models maximum capacity being 1022.
        nogpu: bool.
            Allows to use the GPU is CUDA is available. 'True' ignores the GPU.
        warm_start: bool.
            whether or not to start the extraction where it has stopped. 
            It is usefull in case the processing crashes for any reasons, 
            to avoid having to processing again what have already been processing. 
            However, configuration parameters should NOT be changed in between.
        """        
        self.embs_model = embs_model
        self.extract = extract
        self.log_transform = log_transform

        self.max_num_embs_per_file = max_num_embs_per_file
        self.toks_per_batch = toks_per_batch
        self.truncation_seq_length = truncation_seq_length
        self.nogpu = nogpu
        self.warm_start = warm_start

        self.output_file = None

class EmbeddingsProcessor(EmbeddingsProcessorConfig):
    def __init__(self, fasta_file:str|Path, result_dir:str|Path, config: EmbeddingsProcessorConfig) -> None:
        self.__dict__.update(config.__dict__)

        embeds_dir = result_dir.joinpath('embeddings_' + Path(self.embs_model).stem)
        embs_name  = Path(fasta_file).stem + f"__{Path(self.embs_model).stem}__embs.pt"
        embeds_dir.mkdir(exist_ok=True)

        self.fasta_file  = Path(fasta_file)
        self.output_file = embeds_dir.joinpath(embs_name)

    def get_labels(self, sequence_labels) -> list:
        def format_label(label: str):
            label = label.split('|')
            organism = label[0].split('-')[0]
            mutation = label[1]
            return organism + '_' + mutation
        def remove_comment(label: str):
            idx = label.rfind('|')
            if ' ' in label[idx:]:
                idx = label[idx:].index(' ') + idx 
                label = label[:idx]   
            return label

        if isinstance(sequence_labels, str):
            return format_label(remove_comment(sequence_labels)) 

        for i, label in enumerate(sequence_labels):
            sequence_labels[i] = format_label(remove_comment(label))

        return sequence_labels

    def extract_repr(self):
        model, alphabet = pretrained.load_model_and_alphabet(self.embs_model)
        model.eval()
        if isinstance(model, MSATransformer):
            raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
        if torch.cuda.is_available() and not self.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")

        dataset = FastaBatchedDataset.from_file(self.fasta_file)
        dataset.sequence_labels = self.get_labels(dataset.sequence_labels)
        batches = dataset.get_batch_indices(self.toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(self.truncation_seq_length), batch_sampler=batches)
        print(f"Read {self.fasta_file} with {len(dataset)} sequences")

        with torch.no_grad():
            tmp_batches_d = Path(self.output_file).parent.joinpath('tmp_embs_extract')
            tmp_batches_d.mkdir(exist_ok=self.warm_start)
            tmp_batches_f = [f.name for f in tmp_batches_d.iterdir() if f.is_file()]

            all_batch_file = []
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")

                batch_file = f'batch_{batch_idx}.pt'
                if batch_file in tmp_batches_f:
                    print(f'batch {batch_idx} already processed')
                    all_batch_file.append(Path(tmp_batches_d).joinpath(batch_file))
                    continue

                if torch.cuda.is_available() and not self.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)
                out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
                reprs = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                result = {}
                for i, label in enumerate(labels):
                    truncate_len = min(self.truncation_seq_length, len(strs[i]))
                    result[label] = {layer: t[i,1:truncate_len+1].mean(0).clone() for layer, t in reprs.items()}

                all_batch_file.append(Path(tmp_batches_d).joinpath(batch_file))
                torch.save(result, all_batch_file[-1])

            final_results, file_count = {}, 0
            out_path = Path(self.output_file).with_suffix('')
            max_len = self.max_num_embs_per_file - len(torch.load(all_batch_file[0]))

            print(f'gathering embeddings into:\n- {Path(out_path)}\n')

            all_saved = True
            for i, batch_file in enumerate(all_batch_file):
                current_result = torch.load(batch_file)
                final_results.update(current_result)
                
                if len(final_results) >= max_len or i == len(all_batch_file) - 1:
                    file_to_save = str(out_path) + f"_{file_count}" + ".pt"
                    torch.save(final_results, file_to_save)
                    file_count += 1
                    final_results = {}

                    save_msg = f'> {Path(file_to_save).name}'
                    if Path(file_to_save).is_file() == False:
                        save_msg += 'failed'
                        all_saved = False
                    else:
                        save_msg += ' saved\n'
                    print(save_msg)

            if all_saved:
                os.chmod(tmp_batches_d, 0o777)
                shutil.rmtree(tmp_batches_d)
            else:
                print('embeddings gathering failed')
                print(f'individual embeddings available in:\n- {Path(tmp_batches_d)}\n')

    def load(self):
        def get_info_from_header(header):
            organism, mutation, fitness = header.split('|')
            organism = organism.split('-')[0]
            fitness = float(fitness)
            label = self.get_labels(header)
            return organism, mutation, fitness, label

        embs_files, num = [], []
        for obj in self.output_file.parent.iterdir():
            if obj.is_file() == False:
                continue
            if str(obj.stem[-1]).isnumeric():
                if obj.stem[:-1] == self.output_file.stem + '_':
                    num.append(int(obj.stem[-1]))
                    embs_files.append(obj)
        
        seq_repr = {}
        for file in [embs_files[n] for n in np.argsort(num)]:
            seq_repr.update(torch.load(file))

        embs = {}
        wt_log_transform = {}
        for header, _seq in esm.data.read_fasta(str(self.fasta_file)):
            organism, mutation, fitness, label = get_info_from_header(header)

            if organism not in embs:
                embs[organism] = {'wt': None, 'variant': [], 'fitness': []}

            nums = seq_repr[label].keys()
            repr = seq_repr[label][list(nums)[0]]
            repr = [float(r) for r in repr]

            if mutation.lower() == 'wt':
                if embs[organism]['wt'] is not None:
                    raise ValueError(f'duplicated wild type {organism}')
                embs[organism]['wt'] = repr
                if self.log_transform:
                    wt_log_transform[organism] = np.log10(fitness)                
            else:
                embs[organism]['variant'].append(repr)
                embs[organism]['fitness'].append(fitness)
        
        for org in wt_log_transform:
            log_fit = np.log10(embs[org]['fitness']) - wt_log_transform[org]
            embs[org]['fitness'] = log_fit.tolist()                

        return embs

    def translocate(self, target: str, homologs: list, embs: dict):
        embs = {p: embs[p] for p in [target] + homologs}
        for org in embs:
            wt = embs[org]['wt']
            transl_wt  = np.array(embs[org]['wt']) - wt
            transl_var = np.array(embs[org]['variant']) - wt
            embs[org]['wt'] = transl_wt.tolist()
            embs[org]['variant'] = transl_var.tolist()
        return embs

class DatasetBuilderConfig():
    """
    Configuration of the dataset builder

    Parameters
    ----------
    train_size: float.
        size of the target data used for training
    data_max_len: int.
        number of instances of the addtional data to use. It is usefull if
        the additional data, to be translocated, are big. It allows to repeatable
        and randomly sample 'data_max_len' number of instances at each outer loop
        of the nested cross-validation. 'None' means all additional data are used.
    random_seed: int.
        allows to seed the random states, to get repeatable results.

    """    
    def __init__(self, train_size:float=0.8, data_max_len:int=None, random_seed:int=None) -> None:
        self.train_size = train_size
        if data_max_len is None:
            data_max_len = -1
        self.data_max_len = data_max_len
        self.random_seeds = random_seed

class DatasetBuilder(DatasetBuilderConfig):
    def __init__(self, config: DatasetBuilderConfig) -> None:
        self.__dict__.update(config.__dict__)

        self.wt_fitness = 0
        self.current_seed = 0

    def target_nth_train_test_split(self, X, n) -> tuple[list, list]:
        lens = [int(self.data_max_len), len(X)]
        sub_len = min([l for l in lens if l > 0])
        target_idx = list(range(len(X)))
        
        train_len = int(self.train_size * sub_len)
        test_len  = sub_len - train_len
 
        n_data = len(target_idx)
        start = n * test_len % n_data

        if n % np.ceil(n_data/test_len) == 0:
            if self.random_seeds is not None:
                np.random.seed(n + self.random_seeds)
            self.current_seed = np.random.randint(0, 10000)
        np.random.seed(self.current_seed)
        np.random.shuffle(target_idx)

        current_idx = [target_idx[(start + i) % n_data] for i in range(sub_len)]

        train_idx = current_idx[:train_len]
        test_idx  = current_idx[train_len:]

        return train_idx, test_idx

    def homologs_nth_subsample(self, homologs_types, n) -> list:
        homologs_idxs = {}
        for i, hom_type in enumerate(homologs_types):
            if hom_type not in homologs_idxs:
                homologs_idxs[hom_type] = []
            homologs_idxs[hom_type].append(i)
        
        current_idxs = []
        for hom_type, hom_idxs in homologs_idxs.items():
            lens = [int(self.data_max_len), len(hom_idxs)]
            sub_len = min([l for l in lens if l > 0])
            
            n_data = len(hom_idxs)
            start = n * sub_len % n_data

            if n % np.ceil(n_data/sub_len) == 0:
                if self.random_seeds is not None:
                    np.random.seed(n + self.random_seeds)
                self.current_seed = np.random.randint(0, 1000)
            np.random.seed(self.current_seed)
            np.random.shuffle(hom_idxs)

            current_idxs += [hom_idxs[(start + i) % n_data] for i in range(sub_len)]
            
        return current_idxs

    def get_nth_dataset(self, n:int, embs:dict, target:str, homologs:list=[], standardize:bool=True):
        Xt, yt, Xh, yh, th = [], [], [], [], []
        wtt = embs[target]['wt']
        for prot in [target] + homologs:
            for j in range(len(embs[prot]['variant'])):
                variant = embs[prot]["variant"][j]
                fitness = embs[prot]["fitness"][j]
                if prot == target:
                    Xt.append(variant)
                    yt.append(fitness)
                else:
                    Xh.append(variant)
                    yh.append(fitness)
                    th.append(prot)

        ta_idxs = self.target_nth_train_test_split(Xt, n)
        ho_idxs = self.homologs_nth_subsample(th, n)
        tr_idxs, te_idxs = ta_idxs

        Xt_test, Xt_train = [Xt[i] for i in te_idxs], [Xt[i] for i in tr_idxs] + [wtt]
        yt_test, yt_train = [yt[i] for i in te_idxs], [yt[i] for i in tr_idxs] + [self.wt_fitness]
        Xh_sub, yh_sub    = [Xh[i] for i in ho_idxs], [yh[i] for i in ho_idxs]

        X_train = Xt_train + Xh_sub
        y_train = yt_train + yh_sub
        idxs = np.arange(len(X_train))
        np.random.seed(42)
        np.random.shuffle(idxs)

        X_train = [X_train[i] for i in idxs]
        y_train = [y_train[i] for i in idxs]

        if standardize:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train).tolist()
            Xt_test = scaler.transform(Xt_test).tolist()

        return X_train, y_train, Xt_test, yt_test


from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

class RegressionModelConfig():
    def __init__(self, model_type:str='svr', n_splits:int=5, n_repeats:int=1, parameter_grid:dict|list=None, optimize_once:bool=False, random_seed:int=None) -> None:
        """
        Configuration of the regression model used for fitness prediction.

        Parameters
        ----------
        model_type: str.
            regression model used for fitness prediction. 'svr' or 'lasso' are available
        n_splits: int.
            number of fold used in the inner cross-validation for hyperparameters optimizations.
            Unless 'optimize_once' is set to 'True', it is performed for each outer loop of the 
            nested cross-validation. A higher number can significantly increase the processing time.
        n_repeats: int.
            number of time to repeat the inner cross-validation for hyperparameters optimizations.
            the final number of inner loop is therefore 'n_splits'*'n_repeats'. Unless 'optimize_once'
            is set to 'True', it is performed for each outer loop of the nested cross-validation. 
            A higher number can significantly increase the processing time.
        parameter_grid: dict|list.
            parameter grid to be used in hyperparameters optimization in the inner loop.
            If 'None, the default grid is used.
        optimize_once: bool.
            When 'True', the hyperparameters are only optimized in the first outer loop of the
            nested cross-validation, for each combinations. The resulting hyperparameters are 
            then used for subsequent loops with the corresponding combinations. This is usefull
            to save time an try configurations. However, while this should provide valid results,
            the final processing should be done with 'optimize_once' to False, to better fit the data.
        rand_seed: int.
            allows to seed the random states, to get repeatable results.
        """
        self.model_type = model_type.lower()
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.parameter_grid = parameter_grid
        self.optimize_once = optimize_once          
        self.random_seeds = random_seed

class RegressionModel(RegressionModelConfig):
    def __init__(self, config: RegressionModelConfig) -> None:
        self.__dict__.update(config.__dict__)

        self.lasso_max_iter = 200000

        self.optimized = False
        self.regressor_params = {}
    
    def get_default_grid(self):
        if self.model_type == 'lasso':
            return {
                "alpha": [0.001, 0.005, 0.01]
            }
        elif self.model_type == 'svr':
            return [
                {'kernel': ['rbf'], 'gamma': ['scale', 'auto'], 'C': [0.5, 1, 5, 10, 20, 40, 50, 75]},
                {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto'], 'C': [0.5, 1, 5, 10, 20, 40, 50, 75]}
            ]

    def get_regressor(self):
        if self.model_type == 'lasso':
            return Lasso(**self.regressor_params, max_iter=self.lasso_max_iter)
        elif self.model_type == 'svr':
            return SVR(**self.regressor_params)

    def grid_search_cross_validation(self, X, y, n_jobs):
        if self.optimized == True:
            print(f"- gris search inner cv skipped (optimize_one={self.optimize_once})")
            return
        print(f"- grid search inner cv (regressor={self.model_type}, n_splits={self.n_splits}, n_repeats={self.n_repeats})")

        if self.parameter_grid is None:
            self.parameter_grid = self.get_default_grid()

        regressor  = self.get_regressor()
        RepeaKFold = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_seeds)
        lasso_grid = GridSearchCV(estimator=regressor, param_grid=self.parameter_grid, cv=RepeaKFold, n_jobs=n_jobs)

        self.regressor_params = lasso_grid.fit(X, y).best_params_
        print(f"- best params:" + ", ".join([f"{p}: {v}" for p, v in self.regressor_params.items()]))

        if self.optimize_once:
            self.optimized = True

    def fit_pred(self, X_train, y_train, X_test):
        print(f'- evaluating with best model')
        regressor = self.get_regressor()
        regressor.fit(X_train, y_train)  
        return regressor.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        mse  = mean_squared_error(y_true, y_pred)
        r, _ = spearmanr(y_true, y_pred)
        return mse, r
    
class TeeOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.logfile = open(file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

class Utils():
    def __init__(self, save_data:bool=True) -> None:
        self.save_data = save_data
    
    def get_num_jobs(self, n_jobs):
        if n_jobs is not None:
            if n_jobs < 0:
                n_jobs = os.cpu_count() + 1 - n_jobs
            if n_jobs == 0:
                n_jobs = None
        return n_jobs

    def print_ellapsed_time(self, start):
        now = datetime.now()
        ell = now - start
        print(ell, f'({now.hour}h{now.minute})')

    def save_to_file(self, file_path, data):
        if self.save_data:
            print('saving results')
            np.save(file_path, data)
    
    def count_loop(self, N=None):
        if N is None:
            counter = itertools.count
        else:
            counter = lambda: range(N)

        for n in counter():
            print(f"> loop n={n+1}{'' if N is None else f'/{N}'}")
            yield n

    def ljust(self, val, width, fillchar='0', round_int=3):
        if val is None:
            return None
        
        if isinstance(val, str) == False:
            val = str(round(val, round_int))

        return val.ljust(width, fillchar)
