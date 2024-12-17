

from Scripts.runner import Runner
from Scripts.runner import ProtFitTransConfig
from Scripts.runner import DatasetBuilderConfig
from Scripts.runner import RegressionModelConfig
from Scripts.runner import EmbeddingsProcessorConfig

if __name__ == '__main__':

    # fasta_file: fasta file from 'data' folder
    # embs_model: any fair-esm model (https://github.com/facebookresearch/esm)
    # pred_model: SVM, Lasso
    fasta_file = 'IGPS_ss_tm_tt.fasta'
    embs_model = 'esm1v_t33_650M_UR90S_1'
    pred_model = 'svr'

    config_ProtFitTrans = ProtFitTransConfig(
        fasta_file = 'IGPS_ss_tm_tt.fasta',
        target     = 'SsIGPS',
        min_imp    = 0.1,
        alpha      = 0.05,
        homologs   = ['TmIGPS', 'TtIGPS'],
        n_min      = 10,
        n_max      = 100,
        forced_N   = None
    )
    config_embs = EmbeddingsProcessorConfig(
        embs_model = 'esm1v_t33_650M_UR90S_1',
        extract    = False,
    )
    config_dataset = DatasetBuilderConfig(
        random_seed = None
    )
    config_regressor = RegressionModelConfig(
        model_type  = 'svr',
        random_seed = None
    )

    runner = Runner(
        config_ProtFitTrans = config_ProtFitTrans,
        config_embs = config_embs,
        config_dataset = config_dataset,
        config_regressor = config_regressor,
        warm_start = False,
        n_jobs = None,
        save_data = False
    )

    if runner.embeddings.extract:
        runner.extract_embs()
    runner.load_and_translocate_embs()
    runner.evaluate_translocations()

    runner.finalize_and_print_results()