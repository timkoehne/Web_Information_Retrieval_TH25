from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pyterrier as pt
from importlib import reload
from typing import Iterable
import data_cleaning
import os
import shutil
import pandas as pd
pd.set_option('display.max_columns', None)


ensure_pyterrier_is_loaded()


tira = Client()


pt_dataset = pt.get_dataset('irds:ir-lab-wise-2024/subsampled-ms-marco-deep-learning-20241201-training')


reload(data_cleaning)


class DataCleaningIter(Iterable):
    def __init__(self, dataset_iter) -> None:
        self.dataset_iter = iter(dataset_iter)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.dataset_iter)
        item["text"] = data_cleaning.clean_document(item["text"])
        return item


data_cleaning_iter = DataCleaningIter(pt_dataset.get_corpus_iter(verbose=True))

# Index-Pfad
index_path = os.getcwd() + os.sep + "index"


if os.path.exists(index_path):
    shutil.rmtree(index_path)


indexer = pt.IterDictIndexer(
    index_path=index_path,
    meta={'docno': 50, 'text': 4096},
    overwrite=True,
)


index = indexer.index(data_cleaning_iter)

# Initialize the BM25 and DPH retrievers with score transformation
bm25 = pt.BatchRetrieve(index, wmodel="BM25") >> pt.pipelines.PerQueryMaxMinScoreTransformer()
dph = pt.BatchRetrieve(index, wmodel="DPH") >> pt.pipelines.PerQueryMaxMinScoreTransformer()

# Combine the BM25 and DPH models by using a weighted linear combination
linear = 0.75 * bm25 + 0.25 * dph #bestes
linear2 = bm25 + dph
linear3 = 0.5 * bm25 + 0.5 * dph
linear4 = 0.7 * bm25 + 0.3 * dph
linear5 = 0.3 * bm25 + 0.7 * dph
linear6 = 0.8 * bm25 + 0.2 * dph


# Run BM25 on topics 'text' and display the results
run = linear(pt_dataset.get_topics('text'))

# Evaluate the experiment and capture the results
#experiment_results = pt.Experiment([linear],  # List of the models to evaluate
                                 #  pt_dataset.get_topics('text'),
                                 #  pt_dataset.get_qrels(),
                                 #  eval_metrics=["map", "recip_rank", "ndcg_cut_10", "P_1", "P_5", "P_10"])

# Print the evaluation metrics
#print("\nEvaluation Metrics:")
#print(experiment_results)

from tira.third_party_integrations import persist_and_normalize_run
persist_and_normalize_run(
    run,
    # Give your approach a short but descriptive name tag.
    system_name='RankFusion(bm25+dph)',
    default_output='data/runs',
    upload_to_tira=pt_dataset,
)