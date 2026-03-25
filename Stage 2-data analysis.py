import os
import shutil
import json
import re
import numpy as np
import pandas as pd
from preprocessContents import getFolderPath, find_message_files, createChunks
import umap #TODO pip install umap-learn
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import HDBSCAN


def flatten(convo, value="value2"):
    return "/n".join([f"{msg['from']}:{msg[value]}" for msg in convo["conversations"]])


if __name__=="__main__":
    MSG_RE = re.compile(r"^message_(\d+)\_flat.json$")
    path = getFolderPath()
    jsonList = find_message_files(path, MSG_RE)
    compiledData = []
    
    for json_path in jsonList:
        UserName = json_path.parent.name
        filename= json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        compiledData.extend(data)
    compiledData_df = pd.DataFrame(compiledData)
    flattenedConvos= [flatten(convo, value = "value2") for convo in compiledData]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedings = model.encode(flattenedConvos, convert_to_tensor=True, normalize_embeddings=True)
    
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(data)
    hdb = HDBSCAN(copy=True, min_cluster_size=20)
    hdb.fit(embeddings)
    
    labels=np.unique(hdb.labels_).tolist()
    idx=np.arange(len(compiledData))
    sampledConvos = []
    for L in labels:
        cluster_L_idx = idx[hdb.labels_==L]
        cluster_size = len(cluster_L_idx)
        cluster_samples = random.sample(range(cluster_size), 10)
        sampledConvos.extend(compiledData_df.iloc[cluster_samples].tolist())

traningdata = createChunks(sampledConvos, window_size = 15)




    
    
    