import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
class LoggerRecord(object):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)
    def __call__(self, **kwargs):
        if len(kwargs)==0:
            self.get()
        else:
            self.add(**kwargs)
    def _initialize_metric_dict(self):
        return {'pred':[], 'true':[], 'prob':[]}
    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:
                self.samples = {}
                for _k in range(self.k_fold):
                    self.samples[_k] = self._initialize_metric_dict()
            else:
                self.samples[k] = self._initialize_metric_dict()
    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for sample, value in kwargs.items():
                self.samples[sample].append(value)
        else:
            assert k in list(range(self.k_fold))
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)
    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            true = np.concatenate(self.samples['true'])
            pred = np.concatenate(self.samples['pred'])
            prob = np.concatenate(self.samples['prob'])
        else:
            if k is None:
                true, pred, prob = {}, {}, {}
                for k in range(self.k_fold):
                    true[k] = np.concatenate(self.samples[k]['true'])
                    pred[k] = np.concatenate(self.samples[k]['pred'])
                    prob[k] = np.concatenate(self.samples[k]['prob'])
            else:
                true = np.concatenate(self.samples[k]['true'])
                pred = np.concatenate(self.samples[k]['pred'])
                prob = np.concatenate(self.samples[k]['prob'])
        if initialize:
            self.initialize(k)
        return dict(true=true, pred=pred, prob=prob)
    def evaluate(self, k=None, initialize=False, option='mean'):
        samples = self.get(k)
        if self.num_classes==1:
            if not self.k_fold is None and k is None:
                if option=='mean': aggregate = np.mean
                elif option=='std': aggregate = np.std
                else: raise
                explained_var = aggregate([metrics.explained_variance_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                r2 = aggregate([metrics.r2_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                mse = aggregate([metrics.mean_squared_error(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
            else:
                explained_var = metrics.explained_variance_score(samples['true'], samples['pred'])
                r2 = metrics.r2_score(samples['true'], samples['pred'])
                mse = metrics.mean_squared_error(samples['true'], samples['pred'])
            if initialize:
                self.initialize(k)
            return dict(explained_var=explained_var, r2=r2, mse=mse)
        elif self.num_classes>1:
            if not self.k_fold is None and k is None:
                if option=='mean': aggregate = np.mean
                elif option=='std': aggregate = np.std
                else: raise
                accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
                precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
                recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
                roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:,1]) for k in range(self.k_fold)]) if self.num_classes==2 else np.mean([metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(self.k_fold)])
            else:
                accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
                precision = metrics.precision_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
                recall = metrics.recall_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
                roc_auc = metrics.roc_auc_score(samples['true'], samples['prob'][:,1]) if self.num_classes==2 else metrics.roc_auc_score(samples['true'], samples['prob'], average='macro', multi_class='ovr')
            if initialize:
                self.initialize(k)
            return dict(accuracy=accuracy, precision=precision, recall=recall, roc_auc=roc_auc)
        else:
            raise
    def record(self,train_metric,val_metric,train_loss,test_loss,epoch,k,file_path):
        with open(file_path, 'a') as file:
            epoch = '==========fold_{};epoch:{}==========='.format(k,epoch+1)
            Loss = 'loss:{:.4f}/{:.4f} '.format(train_loss,test_loss)
            ACC =  'ACC:{:.4f}/{:.4f} '.format(train_metric['accuracy'],val_metric['accuracy'])
            pre =  'PRE:{:.4f}/{:.4f} '.format(train_metric['precision'],val_metric['precision'])
            recall = 'RECALL:{:.4f}/{:.4f} '.format(train_metric['recall'],val_metric['recall'])
            AUC =  'AUC:{:.4f}/{:.4f} '.format(train_metric['roc_auc'],val_metric['roc_auc'])
            file.write(epoch + '\n')
            file.write(Loss+ACC+pre+recall+AUC+'\n')