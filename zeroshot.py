import copy
import numpy as np
import os
import pandas as pd
from transformers import pipeline
import torch
SEED = 42

class ZSL(object):
    def __init__(self, 
                 labels, 
                 model,
                 n_classes,
                 domain=None,
                 transform_method=None,        
                 SEED=SEED):        
        #path_model = 'mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
        #path_model = 'bart-large-mnli'
        
        #model_id = os.path.join( os.getcwd(), 'texts', 'models', model)
        model_id = os.path.join( os.getcwd(), 'out', model)
        self.labels = copy.deepcopy(labels)
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("zero-shot-classification", model=model_id, device=device, seed=SEED)
        self.SEED = SEED
        
    def fit(self, X,y):
        pass
        
    def predict(self, X):
        pred = self.predict_proba(X)
        return pred.argmax(axis=1)
    
    def predict_proba(self, X):
        df = pd.DataFrame()
        df['sequences'] = copy.deepcopy(X)
        pred = np.zeros((len(X), len(self.labels)))
        p_aux = self.classifier(df['sequences'].tolist(), self.labels, seed=self.SEED)
        for p in p_aux:
            i = np.where(X==p['sequence'])[0]
            for c,s in zip(p['labels'], p['scores']):
                j = np.where(self.labels==c)[0]
                pred[i,j] = s
        return pred