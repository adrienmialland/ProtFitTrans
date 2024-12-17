

from scripts.runner import Runner
from scripts.runner import ProtFitTransConfig
from scripts.runner import DatasetBuilderConfig
from scripts.runner import RegressionModelConfig
from scripts.runner import EmbeddingsProcessorConfig

if __name__ == '__main__':

    config_ProtFitTrans = ProtFitTransConfig(
        fasta_file = 'IGPS_ss_tm_tt.fasta',
        target     = 'SsIGPS',
        min_imp    = 0.1,
        alpha      = 0.05,
        homologs   = ['TmIGPS', 'TtIGPS'],
        N_loop = 50
    )
    config_embs = EmbeddingsProcessorConfig(
        embs_model = 'esm2_t33_650M_UR50D',
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
        save_data = True
    ).launch()