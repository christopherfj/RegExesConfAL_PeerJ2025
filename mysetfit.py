import os
import copy
import gc
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from transformers.trainer_callback import PrinterCallback
from datasets import Dataset
from transformers import set_seed
from os.path import dirname as up

os.environ['WANDB_DISABLED'] = 'true'

SEED = 42

class SETFIT(object):
    def __init__(self,       
    model,     
    n_classes,
    batch_size,
    num_epochs, 
    learning_rate,  
    #domain=None,
    #transform_method=None,   
    SEED=SEED
    ):
        self.N_CLASSES = n_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.SEED = SEED
        
        #path_model = 'bert-base-spanish-wwm-cased-xnli'
        #path_model = 'distilbert-base-es-multilingual-cased'
        #path_model = 'zeroshot_selectra_medium'
        
        #path_model = 'paraphrase-multilingual-mpnet-base-v2'
        #path_model = 'paraphrase-multilingual-MiniLM-L12-v2'
        
        self.model = SetFitModel.from_pretrained(
            #os.path.join(up(up( os.getcwd() )), 'UOH', 'BHI2024', 'out', path_model),
            #os.path.join( os.getcwd(), 'texts', 'models', model),
          	
          	os.path.join( os.getcwd(), 'out', model),
          
            )

    def fit(self, X, y):#, X_val, y_val):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        df['label'] = copy.deepcopy( y )
        train_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        '''
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X_val)
        df['label'] = copy.deepcopy( y_val )
        eval_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        '''

        args = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            loss=CosineSimilarityLoss,
            num_iterations=20,
            head_learning_rate = self.learning_rate,
            seed = self.SEED,
            show_progress_bar=False,
        )
        args.eval_strategy = args.evaluation_strategy

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            #eval_dataset=eval_dataset,
            #metric="accuracy",
        )
        self.trainer.remove_callback(PrinterCallback)

        self.trainer.train()

    def predict(self, X):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        test_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        return self.trainer.model.predict(test_dataset['text']).numpy()

    def predict_proba(self, X):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        test_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        return self.trainer.model.predict_proba(test_dataset['text']).numpy()

