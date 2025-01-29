from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pyterrier as pt
from importlib import reload
from typing import Iterable
import data_cleaning
import os
import shutil
import time
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd
pd.set_option('display.max_columns', None)

# Ensure that PyTerrier is loaded
ensure_pyterrier_is_loaded()

# Initialize Tira client
tira = Client()

# Load the dataset
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

# Create the data cleaning iterator
data_cleaning_iter = DataCleaningIter(pt_dataset.get_corpus_iter(verbose=True))

# Index path
index_path = os.getcwd() + os.sep + "index"

# Delete the index folder with retry mechanism
def delete_with_retry(path, retries=3, delay=1):
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            print(f"Index folder {path} successfully deleted.")
            break
        except PermissionError:
            print(f"Permission error while deleting {path}. Retrying...")
            time.sleep(delay)  # Wait for a second before retrying
        except Exception as e:
            print(f"Error deleting {path}: {e}")
            break

# Delete the index folder if it exists
if os.path.exists(index_path):
    delete_with_retry(index_path)

# Initialize the indexer
indexer = pt.IterDictIndexer(
    index_path=index_path,
    meta={'docno': 50, 'text': 4096},
    overwrite=True,
)


index = indexer.index(data_cleaning_iter)

# Initialize the BM25 retriever
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=200)  # Return only the top 50 results


monoT5 = MonoT5ReRanker()  # Load the default model: castorini/monot5-base-msmarco


bm25_results = bm25(pt_dataset.get_topics('text'))


mono_pipeline = bm25 >> pt.text.get_text(pt_dataset, "text") >> monoT5

#run = mono_pipeline(pt_dataset.get_topics('text'))


experiment_results = pt.Experiment([mono_pipeline],
                                   pt_dataset.get_topics('text'),
                                   pt_dataset.get_qrels(),
                                   eval_metrics=["map", "recip_rank", "ndcg_cut_10", "P_1", "P_5", "P_10"]
                                   )


print("\nEvaluation Metrics:")
print(experiment_results)

#from tira.third_party_integrations import persist_and_normalize_run
#persist_and_normalize_run(
    #run,
    # Give your approach a short but descriptive name tag.
    #system_name='monoT5ReRank',
    #default_output='data/runs',
    #upload_to_tira=pt_dataset,
#)