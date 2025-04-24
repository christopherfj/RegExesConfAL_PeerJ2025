from utils import *

class Texts(Dataset):
    def __init__(self, texts, labels, max_len, path_model, cased, bert_type):
        self.texts = copy.deepcopy(texts)
        self.labels = copy.deepcopy(labels)
        if bert_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(path_model, do_lower_case=cased)
        elif bert_type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(path_model, do_lower_case=cased)
        elif bert_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(path_model, do_lower_case=cased)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts = [self.texts[i] for i in idx] if isinstance(idx, list) else self.texts[idx]
        labels = [self.labels[i] for i in idx] if isinstance(idx, list) else self.labels[idx]
        
        encodings = [self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        ) for text in texts]

        batch = {
            'label': torch.tensor(labels, dtype=torch.long),
            'input_ids': torch.stack([encoding['input_ids'].flatten() for encoding in encodings]),
            'attention_mask': torch.stack([encoding['attention_mask'].flatten() for encoding in encodings])
        }
        
        return batch


class BertClassifier(nn.Module):
    def __init__(self, n_classes, dropout, path_model, bert_type):
        super(BertClassifier, self).__init__()

        self.bert_type = bert_type
        if bert_type=='bert':
            self.bert = BertModel.from_pretrained(path_model)  
            hdim = 768
        elif bert_type=='distilbert':
            self.bert = DistilBertModel.from_pretrained(path_model)
            hdim = 768
        elif bert_type=='albert':
            self.bert = AlbertModel.from_pretrained(path_model)  
            hdim = 312
        
        self.fc = nn.Sequential(
          						nn.Dropout(dropout),
                                nn.Linear(hdim, n_classes)
                               )
    def forward(self, ids, mask):

        if self.bert_type == 'bert':
            _, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
        elif self.bert_type == 'distilbert':
            pooled_output = self.bert(ids, attention_mask = mask)
            pooled_output= pooled_output[0][:,0]
        elif self.bert_type == 'albert':
            pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
            pooled_output= pooled_output[0][:,0]

        output = self.fc(pooled_output)
        return output

class BERT(object):
    def __init__(
        self,
        n_classes,
        scheduler_opt,
        early_stopping,
        validation_split,
        val_loss_min,
        patience,
        batch_size,
        epochs,
        dropout,
        MAX_SENT_LEN,
        lr,
        RUNS,
        bert_type,
        SEED=SEED
    ):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(self.device)

        self.scheduler_opt = scheduler_opt

        if bert_type == 'albert':
            model_bert = "albert-tiny-spanish"
        elif bert_type == 'distilbert':
            model_bert = "distilbert-base-multilingual-cased"
        elif bert_type == 'bert':
            model_bert = "bert-base-multilingual-uncased"

        self.path_model = os.path.join(
            #os.getcwd().replace('CREGEX', 'FREGEX'), "out", model_bert
            #up( os.getcwd() ), 'FREGEX', "out", model_bert
            os.getcwd(), "out", model_bert
        )
        self.cased = "uncased" in self.path_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.val_loss_min = val_loss_min
        self.patience = patience
        self.dropout = dropout
        self.MAX_SENT_LEN = MAX_SENT_LEN
        self.lr = lr
        self.RUNS = RUNS
        self.n_classes = n_classes
        self.SEED = SEED
        self.gpu = 'cuda:0'
        self.bert_type = bert_type

        #gc.collect()
        #torch.cuda.empty_cache()

    def reset_linear(self, m):
        if type(m) == nn.Linear:
            m.reset_parameters()

    def fit(self, X, y):

        #torch.cuda.empty_cache()
        #gc.collect()

        #print('fitting...')
        
        if self.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_split,
                shuffle=False,
                random_state=self.SEED,
            )
        else:
            X_train = copy.deepcopy(X)
            y_train = copy.deepcopy(y)

        train_data = Texts(X_train, y_train, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        if self.validation_split > 0:
            val_data = Texts(X_val, y_val, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
            val_sampler = RandomSampler(val_data)
            val_dataloader = DataLoader(
                val_data,
                sampler=val_sampler,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )

        self.clf = BertClassifier(self.n_classes, self.dropout, self.path_model, self.bert_type)

        optimizer = Adam(self.clf.parameters(), lr=self.lr)
        self.clf.to(self.device)

        #torch.cuda.empty_cache()
        #gc.collect()

        if self.scheduler_opt:
            total_steps = len(train_dataloader) * self.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
            #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=2000)

        epochs_stop = 0
        self.loss_training = []
        self.loss_val = []
        
        fcn = nn.CrossEntropyLoss()

        #scaler = torch.cuda.amp.GradScaler() ##new


        for epoch_i in range(0, self.epochs):
            train_loss = 0
            self.clf.train()
            for step, batch in enumerate(train_dataloader):

                gc.collect()
                #torch.cuda.empty_cache()
                with torch.cuda.device(self.gpu):
                    torch.cuda.empty_cache()
                
                optimizer.zero_grad() #

                labels = batch['label'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                input_masks = batch['attention_mask'].to(self.device)
                

                #optimizer.zero_grad()
                with torch.cuda.amp.autocast():###new
                    
                    logits = self.clf(input_ids, input_masks)

                    batch_loss = fcn(logits.view(-1, self.n_classes), labels.view(-1))

                del input_ids
                del input_masks
                del labels
                #gc.collect()
                #torch.cuda.empty_cache()

                train_loss += batch_loss.item()
                
            
                batch_loss.backward()
                
                
                if self.scheduler_opt:
                    clip_grad_norm_(parameters=self.clf.parameters(), max_norm=1.0)

                
                optimizer.step()

                if self.scheduler_opt:
                    scheduler.step()
                
                        
                
                #scaler.scale(batch_loss).backward()
                #scaler.step(optimizer)
                #scaler.update()
                

            train_loss /= len(train_dataloader.dataset)
            self.loss_training.append(train_loss)

            if self.validation_split > 0:
                val_loss = 0
                self.clf.eval()
                for step, batch in enumerate(val_dataloader):

                    gc.collect()
                    with torch.cuda.device(self.gpu):
                        torch.cuda.empty_cache()

                    labels = batch['label'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    input_masks = batch['attention_mask'].to(self.device)

                    logits = self.clf(input_ids, input_masks)
                    batch_loss = fcn(
                        logits.view(-1, self.n_classes), labels.view(-1)
                    )

                    del input_ids
                    del input_masks
                    del labels
                    #gc.collect()
                    #torch.cuda.empty_cache()

                    val_loss += batch_loss.item()

                val_loss /= len(val_dataloader.dataset)
                self.loss_val.append(val_loss)

                if self.early_stopping:
                    if val_loss < self.val_loss_min:
                        self.val_loss_min = val_loss
                        epochs_stop = 0
                        params_model = copy.deepcopy(self.clf.state_dict())
                    else:
                        epochs_stop += 1
                    if epochs_stop >= self.patience:
                        self.clf.load_state_dict(params_model)
                        #print(epoch_i, "epochs")
                        break

        del train_dataloader

        if self.validation_split > 0:
            del val_dataloader
        #gc.collect()
        #torch.cuda.empty_cache()

    def predict(self, X_test):

        #gc.collect()
        #torch.cuda.empty_cache()

        print('predicting...')
        # fake labels
        y = np.zeros(len(X_test))
        prediction_data = Texts(X_test, y, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        prediction_sampler = SequentialSampler(prediction_data)
        test_dataloader = DataLoader(
            prediction_data,
            sampler=prediction_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        #gc.collect()
        #torch.cuda.empty_cache()

        self.clf.eval()
        predictions = []
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):

                gc.collect()
                #torch.cuda.empty_cache()
                with torch.cuda.device(self.gpu):
                    torch.cuda.empty_cache()

                labels = batch['label'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                input_masks = batch['attention_mask'].to(self.device)
                
                logits = self.clf(input_ids, input_masks)

                del input_ids
                del input_masks
                del labels
                #gc.collect()
                #torch.cuda.empty_cache()

                logits = F.softmax(logits, dim=1)
                logits = logits.detach().cpu().numpy()
                predictions += list(np.argmax(logits, axis=1))

        return np.array(predictions, dtype=int)

    def apply_dropout(self, m):
        if type(m) == nn.Dropout:
            m.train()

    def predict_proba(self, X_u):

        #gc.collect()
        #torch.cuda.empty_cache()

        # fake labels
        y = np.zeros(len(X_u))
        prediction_data = Texts(X_u, y, self.MAX_SENT_LEN, self.path_model, self.cased, self.bert_type)
        prediction_sampler = SequentialSampler(prediction_data)
        test_dataloader = DataLoader(
            prediction_data,
            sampler=prediction_sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.clf.eval()
        self.clf.apply(self.apply_dropout)
        probs = []
        for times in range(self.RUNS):

            logits_sum = np.array([])
            with torch.no_grad():
                for step, batch in enumerate(test_dataloader):

                    gc.collect()
                    #torch.cuda.empty_cache()
                    with torch.cuda.device(self.gpu):
                        torch.cuda.empty_cache()

                    labels = batch['label'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    input_masks = batch['attention_mask'].to(self.device)

                    logits = self.clf(input_ids, input_masks)

                    del input_ids
                    del input_masks
                    del labels
                    #gc.collect()
                    #torch.cuda.empty_cache()

                    logits = F.softmax(logits, dim=1)
                    logits = logits.detach().cpu().numpy()
                    if len(logits_sum) == 0:
                        logits_sum = copy.deepcopy(logits)
                    else:
                        logits_sum = np.vstack((logits_sum, logits))
            probs.append(logits_sum)
        probs = np.mean(probs, axis=0)
        return probs
