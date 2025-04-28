from utils import *
from fregex import FREGEX
from bert import BERT
from mysetfit import SETFIT
from zeroshot import ZSL

class CREGEX(object):
    def __init__(self, 
                FILENAME, MODEL_NAMES, N_CLASSES, THR_CONF_CLF_opt = True,
                PROBS_THR=np.arange(0.50, 1, 0.05), CLFS = 1,
                NGRAM_MIN=NGRAM_MIN, pnumbers=pnumbers, 
                gap_cmb=gap_cmb, whitespaces=whitespaces, lexicon=lexicon, HYPERPARAMS=HYPERPARAMS, SEED=SEED):
        self.__metaclass__ = 'CREGEX'
        self.FILENAME = FILENAME
        clfs, _ = MODEL_NAMES.split('*')
        clfs = clfs.split('.')
        self.MODEL_NAMES = [clf for clf in clfs]
        self.N_CLASSES = N_CLASSES
        self.NGRAM_MIN = NGRAM_MIN
        self.pnumbers=pnumbers
        self.gap_cmb = gap_cmb
        self.whitespaces = whitespaces
        self.lexicon = lexicon
        self.SEED = SEED
        self.HYPERPARAMS = HYPERPARAMS
        self.regexes = {}
        self.labeled_regexes = {}
        self.kw = []
        self.distribution = defaultdict(list)
        self.y = defaultdict(list)
        self.rndm = np.random.RandomState(self.SEED)
        self.models = {}
        self.THR_CONF_CLF_opt = THR_CONF_CLF_opt
        self.THR_CONF_CLFS = {}
        self.PROBS_THR = PROBS_THR
        self.CLFS = CLFS
        self.times = defaultdict(float)

    def fit(self, X,y, X_val, y_val):

        self.regexes = {}
        self.labeled_regexes = {}
        self.kw = []
        self.distribution = defaultdict(list)
        self.y = defaultdict(list)
        self.models = {}

        print('CREGEX...fit')
        st = time.time()
        fregex = FREGEX(X, y, self.FILENAME)
        fregex.fit()
        self.regexes.update( fregex.transform() )        
        self.kw = copy.deepcopy(self.lexicon[self.FILENAME])
        self.pattern2token = copy.deepcopy( fregex.pattern2token )
        self.token2pattern = copy.deepcopy( fregex.token2pattern )
        self.tokens2pos = copy.deepcopy( fregex.tokens2pos )
        self.stopwords = copy.deepcopy(fregex.stopwords)
        self.regexes, self.regex2class = get_classes_regexes(self.regexes, y, self.tokens2pos)
        labeled_regexes, labeled_regexes_filtered, labeled_regexes_all = get_filtered_regexes(self.regexes, y, self.kw, self.pattern2token, self.regex2class )
        self.labeled_regexes.update( labeled_regexes )
        del labeled_regexes
        del labeled_regexes_filtered
        del labeled_regexes_all
        gc.collect()

        keys = copy.deepcopy( list( self.regexes.keys() ) )
        for key in keys:
            if key not in self.labeled_regexes:
                self.regexes.pop(key)
                self.regex2class.pop(key)

        self.times['train-'+'cregex'] = time.time()-st
        del st
        gc.collect()

        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME+'...fit')
            tokens = None
            opt = None
            regexes_aux = None
            model = None
            X_val_aux = None
            y_val_aux = None
            X_l_aux = None
            y_l_aux = None
            if 'random' not in MODEL_NAME: #clf            
                seed_everything()
                #if 'bert' not in MODEL_NAME:
                if 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME and 'zsl' not in MODEL_NAME:
                    st = time.time()    
                    _, NGRAM_SIZE = MODEL_NAME.split('-')
                    NGRAM_SIZE = int(NGRAM_SIZE.replace('n',''))
                    tokens = n_grams(X, NGRAM_SIZE)
                    opt = False
                    regexes_aux = {}
                    y_l_aux = copy.deepcopy(y)                    
                    X_l_aux = copy.deepcopy( get_matrix(tokens, X, regexes_aux, opt) )
                    X_val_aux = copy.deepcopy( get_matrix(tokens, X_val, regexes_aux, opt) )
                    y_val_aux = copy.deepcopy(y_val)
                    X_train_val = copy.deepcopy( np.vstack((X_l_aux, X_val_aux)) )            
                    ps = PredefinedSplit( np.array( [0]*len(y)+[-1]*len(y_val) ) )
                    y_train_val = copy.deepcopy( np.hstack((y, y_val_aux)) )
                    HYPERPARAMS = best_model(MODEL_NAME, ps, X_train_val, y_train_val)
                    model = select_trad_model(MODEL_NAME, HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME] = copy.deepcopy(HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME+'-cregex'] = copy.deepcopy(HYPERPARAMS)
                    model.fit(X_l_aux, y_l_aux)
                    self.times['train-'+MODEL_NAME] = time.time()-st
                    #self.times['train-'+'cregex'] += self.times['train-'+MODEL_NAME]
                    del st
                    gc.collect()
                else:
                    st = time.time()  
                    X_l_aux = copy.deepcopy(X)
                    y_l_aux = copy.deepcopy(y)
                    X_val_aux = copy.deepcopy(X_val)
                    y_val_aux = copy.deepcopy(y_val)
                    if 'bert' in MODEL_NAME:
                        model = BERT(**self.HYPERPARAMS['bert'])
                        model.fit(X_l_aux, y_l_aux)
                    elif 'setfit' in MODEL_NAME:
                        model = SETFIT(**self.HYPERPARAMS['setfit'])
                        model.fit(X_l_aux, y_l_aux) #, X_val_aux, y_val_aux)
                    elif 'zsl' in MODEL_NAME:
                        model = ZSL(**self.HYPERPARAMS['zsl'])
                        model.fit(X_l_aux, y_l_aux) #, X_val_aux, y_val_aux)
                    self.times['train-'+MODEL_NAME] = time.time()-st
                    del st
                    gc.collect()
                #model.fit(X_l_aux, y_l_aux)
                pred_val = model.predict_proba(X_val_aux)
            else:
                st = time.time()
                X_val_aux = copy.deepcopy(X_val)
                y_val_aux = copy.deepcopy(y_val)
                seed_everything()
                pred_val = []
                for _ in range(len(X_val_aux)):
                    pred = self.rndm.randint(0, self.N_CLASSES) 
                    pond = 1/(self.N_CLASSES+1)
                    preds = np.ones(self.N_CLASSES)*pond
                    preds[pred] = 1-pond*(self.N_CLASSES-1)
                    pred_val.append(preds)
                self.times['train-'+MODEL_NAME] = time.time()-st
                del st
                gc.collect()
                pred_val = np.array(pred_val)

            if self.THR_CONF_CLF_opt:
                precision, recall, thresholds, weights = prec_rec_curves(y_val_aux, pred_val, self.PROBS_THR, self.N_CLASSES)
                if self.N_CLASSES<3:
                    precision[np.isnan(precision)] = 0
                    recall[np.isnan(recall)] = 0
                    fscore = (2 * precision * recall) / (precision + recall)
                    fscore[np.isnan(fscore)] = 0
                    idx = np.argmax(fscore)
                    self.THR_CONF_CLFS[MODEL_NAME] = thresholds[idx]
                else:
                    aux_thr = []
                    for c in range(self.N_CLASSES):
                        precision[c][np.isnan(precision[c])] = 0
                        recall[c][np.isnan(recall[c])] = 0
                        fscore = (2 * precision[c] * recall[c]) / (precision[c] + recall[c])
                        fscore[np.isnan(fscore)] = 0
                        idx = np.argmax(fscore)
                        aux_thr.append(thresholds[c][idx])
                    self.THR_CONF_CLFS[MODEL_NAME] = aux_thr

                self.distribution['thresholds-'+MODEL_NAME] = [precision, recall, fscore, thresholds, weights] 
                #self.THR_CONF_CLF = thresholds[idx]
                #print('THR_CONF_CLF', self.THR_CONF_CLF)
                #fig = pylab.figure(1)
                #pylab.plot(recall, precision)
                #pylab.show()
                #print('idx', precision[idx], recall[idx], fscore[idx])

            self.models[MODEL_NAME] = [tokens, opt, regexes_aux, model, X_l_aux, y_l_aux]            
            
    def predict(self, X): #always predict_proba

        self.y = defaultdict(list)

        for MODEL_NAME in self.MODEL_NAMES:
            st = time.time()
            print(MODEL_NAME+'...predict')
            tokens, opt, regexes_aux, model, X_train, y_train = self.models[MODEL_NAME]
            if 'random' not in MODEL_NAME: #-clf
                #if 'bert' not in MODEL_NAME:
                if 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME and 'zsl' not in MODEL_NAME:
                    X_test_aux = copy.deepcopy( get_matrix(tokens, X, regexes_aux, opt) )
                else:
                    X_test_aux = copy.deepcopy(X)
                predictions = model.predict_proba( X_test_aux )   
     
            elif 'random' in MODEL_NAME:
                seed_everything()
                predictions = []
                for _ in range(len(X)):
                    pred = self.rndm.randint(0, self.N_CLASSES) 
                    pond = 1/(self.N_CLASSES+1)
                    preds = np.ones(self.N_CLASSES)*pond
                    preds[pred] = 1-pond*(self.N_CLASSES-1)
                    predictions.append(preds)
                predictions = np.array(predictions)

            self.times['predict-'+MODEL_NAME] = time.time()-st
            del st
            gc.collect()    

            predictions = [list(p) for p in predictions]

            self.y[MODEL_NAME] = copy.deepcopy(predictions) #list(predictions)

        print('CREGEX...predict')
        i = -1     
        for text in X:
            i+=1
            classe = None
            for MODEL_NAME in self.MODEL_NAMES:
                st = time.time()
                if self.N_CLASSES<3:
                    pos_pred = self.y[MODEL_NAME][i][1]
                    THR_PROB = self.THR_CONF_CLFS[MODEL_NAME]
                else:
                    c = np.argmax(self.y[MODEL_NAME][i])
                    pos_pred = self.y[MODEL_NAME][i][c]
                    THR_PROB = self.THR_CONF_CLFS[MODEL_NAME][c]

                if pos_pred>=THR_PROB:
                    classe = copy.deepcopy( self.y[MODEL_NAME][i] )
                    self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('clf-A', None) )

                    self.times['predict-'+MODEL_NAME+'-cregex'] += self.times['predict-'+MODEL_NAME]/len(X)
                    
                else:

                    flag = False
                    labels = []
                    confs = []
                    regexs = []
                    max_conf = []
                    regexs_labels = []
                    rlc = []

                    for regex in self.labeled_regexes:
                        label, conf = self.labeled_regexes[regex]
                        _, numbers_aux, _, _, _ = self.regexes[regex]
                        f = findall(regex, [], numbers_aux, text)
                        
                        if f:
                            flag = True
                            labels.append(label)
                            confs.append(conf)
                            regexs.append(regex)

                    if flag:
                        regexs = np.array(regexs)
                        labels = np.array(labels)
                        confs = np.array(confs)

                        rlc = list(zip(regexs, labels, confs))                
                        eps = 1e+4
                        rlc = sorted( rlc, 
                                key=lambda x:x[2]+len( re.split(r'(?:%s|%s)' %(re.escape(self.gap_cmb), re.escape(self.whitespaces)), x[0]))/eps, 
                                reverse=True)
                        
                        classe = copy.deepcopy( rlc[0][1] )
 
                        max_conf = rlc[0][2]
                        pos_aux = copy.deepcopy(classe)
                        #pond = (1-max_conf)/( predictions.shape[1]-1)
                        pond = (1-max_conf)/( self.N_CLASSES-1)
                        #classe =  np.ones(predictions.shape[1])*pond
                        classe =  np.ones(self.N_CLASSES)*pond
                        classe[ pos_aux ] = max_conf

                        #print(classe, classe.sum())

                        r_aux, l_aux, c_aux = zip(*rlc)

                        self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('rex', [list(r_aux), list(l_aux), list(c_aux)]) )

                        self.times['predict-'+MODEL_NAME+'-cregex'] += time.time()-st
                        #del st
                        #gc.collect()

                    else:
                        classe = copy.deepcopy( self.y[MODEL_NAME][i] )
                        self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('clf-B', None) )

                        self.times['predict-'+MODEL_NAME+'-cregex'] += self.times['predict-'+MODEL_NAME]/len(X)

                classe = list(classe)

                #self.times['predict-'+MODEL_NAME+'-cregex'] += time.time()-st
                del st
                gc.collect()

                self.y[MODEL_NAME+'-cregex'].append( classe )

        for key in self.y:
            self.y[key] = np.array(self.y[key])

        if self.CLFS == 1:
            return self.y[MODEL_NAME+'-cregex']
        else:
            return self.y


    def predict_proba(self, X):
        return self.predict(X)



