from fregex import *
from utils import *
from bert import *
from cregex import *
from mysetfit import SETFIT
from zeroshot import ZSL

def get_tokens(X,y,N, FILENAME):
    regexes = {}
    opt = False
    tokens = []
    if type(N)==int:
        tokens = n_grams(X, N)                
    elif 'regex' in N:
        mode = N.split('-')[0]
        fregex = FREGEX(X,y, FILENAME, mode)
        fregex.fit()
        regexes = copy.deepcopy( fregex.transform() )
        opt = True
        tokens = list( regexes.keys() )        
    tokens = sorted(tokens)
    return regexes, opt, tokens

class Curves(object):
    def __init__(self,
                X_train, y_train,
                X_val, y_val,
                X_test,
                N_CLASSES, CURVE, MODEL,
                BATCH, FILENAME,
                HYPERPARAMS=HYPERPARAMS, SEED = SEED
                ):
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.X_val = copy.deepcopy(X_val)
        self.y_val = copy.deepcopy(y_val)
        self.X_test  = copy.deepcopy(X_test)
        self.N_CLASSES = N_CLASSES
        self.CURVE = CURVE
        self.MODEL = MODEL.split('-')[0] #clf
        self.BATCH = BATCH #lc
        self.FILENAME = FILENAME
        self.SEED = SEED
        self.results = {}
        self.dst_cregex = []
        self.N_FEATURES = []
        self.X_u = []

        if 'cregex' in MODEL:
            split_aux = MODEL.split('-')
            if len(split_aux)>2: #clf-nx-cregex
                self.NGRAM_SIZE = '-'+MODEL.split('-')[1]+'*cregex'
            else: #clf-cregex
                self.NGRAM_SIZE = '*cregex'
        elif 'fregex' in MODEL:
            self.NGRAM_SIZE = 'fregex'
        elif 'bert' not in MODEL and 'setfit' not in MODEL and 'zsl' not in MODEL:
            self.NGRAM_SIZE = int(MODEL[-1])
        elif 'bert' in MODEL:
            self.NGRAM_SIZE = 'bert'
        elif 'setfit' in MODEL:
            self.NGRAM_SIZE = 'setfit'
        elif 'zsl' in MODEL:
            self.NGRAM_SIZE = 'zsl'
            
        if 'bert' not in self.MODEL and 'setfit' not in self.MODEL and 'zsl' not in self.MODEL:
            self.HYPERPARAMS = self.search_hyperparams(self.X_train, self.y_train, self.X_val, self.y_val)
        else:
            self.HYPERPARAMS = copy.deepcopy(HYPERPARAMS[self.MODEL])

        #print('NGRAM_SIZE', self.NGRAM_SIZE)

    def search_hyperparams(self, X_l, y_l, X_val, y_val):
        ps = PredefinedSplit( np.array( [0]*len(y_l)+[-1]*len(y_val) ) )
        y_l_val = copy.deepcopy( np.hstack((y_l, y_val)) )
        regexes, opt, tokens = get_tokens(X_l, y_l, self.NGRAM_SIZE, self.FILENAME)
        X_l_aux = copy.deepcopy( get_matrix(tokens, X_l, regexes, opt) )
        X_val_aux = copy.deepcopy( get_matrix(tokens, X_val, regexes, opt) )
        X_l_val = copy.deepcopy( np.vstack((X_l_aux, X_val_aux)) )
        return best_model(self.MODEL, ps, X_l_val, y_l_val)

    def start(self):
        X_l = np.array([])
        y_l = np.array([])
        classes_ = copy.deepcopy( self.y_train[:self.BATCH] )
        while len(set(classes_)) != self.N_CLASSES:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.SEED)
            classes_ = copy.deepcopy(self.y_train[:self.BATCH])
        del classes_
        gc.collect()
        X_l = self.X_train[:self.BATCH]
        y_l = self.y_train[:self.BATCH]
        X_u = self.X_train[self.BATCH:]
        y_u = self.y_train[self.BATCH:]

        return X_l, y_l, X_u, y_u, [], []

    def model_selection(self, X_train, y_train, X_test, X_u=[], results=False, return_model=False):
        seed_everything()

        X_l_aux = copy.deepcopy( X_train )
        y_l = copy.deepcopy( y_train )
        X_test_aux = copy.deepcopy( X_test )
        X_u_aux = copy.deepcopy(X_u)

        if type(self.NGRAM_SIZE)!=int and 'cregex' in self.NGRAM_SIZE: #cregex
            #div
            if 'bert' not in self.MODEL and 'setfit' not in self.MODEL and 'zsl' not in self.MODEL: #bow-fregex
                NGRAM_SIZE = int( self.NGRAM_SIZE[2] )
                regexes, opt, tokens = get_tokens(X_l_aux, y_l, NGRAM_SIZE, self.FILENAME)
                self.N_FEATURES.append( len(tokens) )
                #self.X_u = copy.deepcopy( get_matrix(tokens, X_u_aux, regexes, opt) )
                del regexes
                del opt
                del tokens
            else:
                self.N_FEATURES.append( 768 )
            #div
            model = CREGEX(self.FILENAME, self.MODEL+self.NGRAM_SIZE, self.N_CLASSES, True) 
            model.fit(X_l_aux, y_l, self.X_val, self.y_val) 
        else:
            if 'bert' not in self.MODEL and 'setfit' not in self.MODEL and 'zsl' not in self.MODEL: #bow-fregex
                model = select_trad_model(self.MODEL, self.HYPERPARAMS)
                regexes, opt, tokens = get_tokens(X_l_aux, y_l, self.NGRAM_SIZE, self.FILENAME)
                X_l_aux = copy.deepcopy( get_matrix(tokens, X_l_aux, regexes, opt) )
                X_test_aux = copy.deepcopy( get_matrix(tokens, X_test_aux, regexes, opt) )
                X_u_aux = copy.deepcopy( get_matrix(tokens, X_u_aux, regexes, opt) )
                model.fit(X_l_aux, y_l)
                self.N_FEATURES.append( X_l_aux.shape[1] )
                #div
                #self.X_u = copy.deepcopy(X_u_aux)
                #div
            else:
                #div
                #regexes, opt, tokens = get_tokens(X_l_aux, y_l, 1, self.FILENAME)
                #self.X_u = copy.deepcopy( get_matrix(tokens, X_u_aux, regexes, opt) )
                #del regexes
                #del opt
                #del tokens
                #div
                if 'bert' in self.MODEL:
                    model = BERT(**self.HYPERPARAMS)
                    model.fit(X_l_aux, y_l)
                    self.N_FEATURES.append( 768 )
                elif 'setfit' in self.MODEL:
                    model = SETFIT(**self.HYPERPARAMS)
                    model.fit(X_l_aux, y_l)#, self.X_val, self.y_val)
                    self.N_FEATURES.append( 768 )
                elif 'zsl' in self.MODEL:
                    model = ZSL(**self.HYPERPARAMS)
                    model.fit(X_l_aux, y_l)#, self.X_val, self.y_val)
                    self.N_FEATURES.append( 768 )
                    
        pred_u = []
        scores_u = []
        pred = model.predict_proba(X_test_aux)
        
        if type(self.NGRAM_SIZE)!=int and 'cregex' in self.NGRAM_SIZE: #cregex
            self.dst_cregex.append( copy.deepcopy(model.distribution) )

        if len(X_u)>0:
            pred_u = model.predict_proba(X_u_aux)
            scores_u = entropy(pred_u, base=2, axis=1)
        del X_l_aux
        del y_l
        del X_test_aux
        gc.collect()
        if return_model:
          	return pred, pred_u, scores_u, model
        else:
            del model
            gc.collect()
            return pred, pred_u, scores_u

    def learningCurve(self):
        MIN_X = 1
        BATCH = 0
        X_BATCH = -1
        scores = []
        y_clf = []
        y_u_dst = []
        X_l, y_l, X_u, y_u, x, y = self.start()
        #div
        #matrix_U = []
        #div

        while len(X_u)>=0:

            print(len(X_u))

            pred, pred_u, scores_u, clf = self.model_selection(X_l, y_l, self.X_test, X_u, False, True)
            x.append( len(y_l) ) 
            y.append( pred ) 
            
            if len(X_u)==0:
                break

            indexes = np.array([], dtype = int)
            if self.CURVE == 'PL':
                indexes = np.arange(len(X_u)) #shuffle(np.arange(len(X_u)), random_state = self.SEED) 
            else:
                indexes = np.argsort( scores_u )[::-1]
                
                scores.append( scores_u[indexes] )
                y_u_dst.append( y_u[indexes] )
                y_clf.append( pred_u[indexes] ) 

                #div
                #matrix_U.append( [ self.X_u[indexes, :], y_u[indexes] ] )
                #div

            X_l = np.concatenate((X_l, X_u[indexes[:self.BATCH]] )) 
            y_l = np.concatenate((y_l, y_u[indexes[:self.BATCH]] ))
            X_u = np.delete(X_u, indexes[:self.BATCH], axis = 0)
            y_u = np.delete(y_u, indexes[:self.BATCH], axis = 0)

            if len(X_u)<=0:
                X_l = copy.deepcopy( self.X_train )
                y_l = copy.deepcopy( self.y_train )
                del self.X_train
                del self.y_train
                gc.collect()
            del clf
            gc.collect()

        self.results['x'] = np.array(x)
        self.results['y'] = np.array(y)
        self.results['scores'] = np.array(scores, dtype=object)
        self.results['y_u_dst'] = np.array(y_u_dst, dtype=object)
        self.results['y_clf'] = np.array(y_clf, dtype=object)
        self.results['dst_cregex'] = np.array(self.dst_cregex, dtype=object)
        #div
        #self.results['X_u-y_u'] = np.array(matrix_U, dtype=object)
        #div

        del X_l
        del y_l
        del X_u
        del y_u
        del self.X_test
        del self.X_val
        del self.y_val
        gc.collect()