import numpy as np
from tqdm import tqdm
import itertools

from numba import jit
import pandas as pd
from sklearn.metrics import label_ranking_loss

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import pairwise_distances

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from scipy import sparse
import copy

x = sparse.load_npz('../TheData/x_norm_sparse.npz')
y = sparse.load_npz('../TheData/y.npz')

x = np.array(x.todense())
y = np.array(y.todense())
allSmiles = pd.read_csv('../TheData/allSmiles.csv', header=None)
targetNames = pd.read_csv('../TheData/targetNames.csv', header=None)

from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

def makeCorrelations(y_in):
    num_lines = len(y_in)
    tot_instances = np.sum(y_in, axis=0)
    L = np.zeros([y_in.shape[1], y_in.shape[1]])

    for row in tqdm(y_in):
        if np.sum(row)>1:
            for j,k in itertools.permutations(np.where(row==1)[0], 2):
                L[j][k] += (1)/(tot_instances[k])

    return L

from numba import jit

@jit(nopython=True)
def fill_in_estimates(new_est_matrix, current_label_matrix, L):
    L1 = 1-L
    wheres = np.where(current_label_matrix==0)
    for j,k in zip(wheres[0], wheres[1]):
        probs = L1[k][current_label_matrix[j]==1]
        probs2 = 1-np.prod(probs)
        new_est_matrix[j][k] = probs2
    return new_est_matrix #return the matrix with only the new labels.

L = makeCorrelations(y)
probabilities_matrix = fill_in_estimates(np.array(y, dtype='float64'),
                                    np.array(y, dtype='float64'),
                                    L)

multiple_labels = np.sum(y,axis=1)>1 # get ligands with more than one label
multiple_labels.nonzero()[0] # this is all those ligands

tot_instances = np.sum(y, axis=0)

def makeNewLabels(probabilities_matrix, threshold):
    prob2 = copy.copy(probabilities_matrix)
    prob2[prob2>=threshold]=1
    prob2[prob2<threshold]=0
    return prob2

y_dict = dict()
for threshold in [0.02, 0.05, 0.2, 0.5, 0.9, 1]:
    y_dict[threshold] = makeNewLabels(probabilities_matrix, threshold)


class VirtualScreeningBootstrapper():
    def __init__(self, x, y_new, y_orig, clf, numNegs=0.2):
        self.x = x
        self.y_new = y_new
        self.y_orig = y_orig
        self.clf = clf
        
        self.positive_indices = None
        self.negative_indices = None
        self.dist = None
        self.targetIndex = None
        self.fraction = None
        self.numNegs = numNegs

    def setTarget(self, target_index):
        self.targetIndex = target_index
        ##We use the Original labels to select a positive. All labels predicted by the correlation graph will be used for training
        self.positive_indices = np.where(self.y_orig[:,self.targetIndex]==1)[0]
        ##We use the New labels to choose negatives - that is, no New label can possibly be selected as a negative. 
        self.negative_indices = np.where(self.y_new[:,self.targetIndex]==0)[0]
        self.weights = np.ones(len(self.positive_indices))

        print(len(self.positive_indices), len(np.where(self.y_new[:,self.targetIndex]==1)[0]))
        self.dist = pairwise_distances(np.array(self.x[self.positive_indices], dtype=bool), metric='dice')
 
    def maskNN(self): #masks the nearest neighbors of a randomly selected positive ligand
        self.weights=normalize([self.weights], norm='l1')[0]
        selection = np.random.choice(self.positive_indices, p=self.weights) #take a random true positive ligand
        ligand_index = np.where(self.positive_indices==selection)[0][0]
        
        knn = np.where(self.dist[ligand_index]<self.fraction)[0]
        add=0.01
        while len(knn)<10: #ensure we have at least 10 ligands in the positives in the test set
            knn = np.where(self.dist[ligand_index]<(self.fraction+add))[0]
            add+=0.01    

        distances = self.dist[ligand_index][knn] 
        self.weights[knn] *= (distances+0.01)

        mask = np.ones(len(self.x), dtype=bool) #create a mask
        mask[self.positive_indices[knn]]=False #set only the neighbors to False
        return mask
     
    def maskNegs(self, mask):
        neg_selection = np.random.choice(self.negative_indices, int(self.numNegs*len(self.negative_indices)))
        mask[neg_selection]=False
        return mask
    
    def evaluateFold(self, clf, mask):
        probs = clf.predict_proba(self.x[~mask])[:,1] #take probability of test ligands being positive
        ##Evaluate on the new labels, but remember only positives and negatives from the old labels have been chosen as test. 
        ##any newly predicted labels are gauranteed to be in the training set. 
        ranking_loss = label_ranking_loss(self.y_new[~mask][:,self.targetIndex].reshape(1,-1), probs.reshape(1,-1))
        return ranking_loss
        
    def bootstrap_par(self, mask):
        clf = copy.copy(self.clf)
        #Train with the NEW, predicted labels. 
        clf.fit(self.x[mask], self.y_new[mask][:,self.targetIndex])
        ranking_loss = self.evaluateFold(clf, mask)
        return ranking_loss
    
    def makeMask(self):
        mask = self.maskNN() #generate a mask for a random block of positives
        mask = self.maskNegs(mask) #mask some negatives too
        return mask
    
    def bootStrap(self, target_index, fraction, repeats, tolerance=0.0025, numJobs=1):
        self.setTarget(target_index)
        self.fraction = fraction

        ranking_losses = list()
        
        for rep in range(repeats):
            print(rep, end='\r')
            if numJobs>1:
                masks = [self.makeMask() for _ in range(numJobs)] #make the masks first to ensure even sampling by the weights.
                                                                ##otherwise you might get odd results from joblib setting weights 
                                                                ##all at the same time
                rlosses = Parallel(n_jobs=numJobs, backend='threading')(delayed(self.bootstrap_par)(mask) for mask in masks)
                for rloss in rlosses:
                    ranking_losses.append(rloss)
            else:
                mask = self.maskNN()
                ranking_loss = self.bootstrap_par(mask)
                ranking_losses.append(ranking_loss)

            if len(ranking_losses)>53:
                medians = [np.median(ranking_losses[:-i]) for i in np.arange(1,51)]
                diffs = [np.abs(medians[-1]- median) for median in medians[:-1]]
                diffs_percent = [(d / medians[-1])*100 for d in diffs]
                if max(diffs_percent)<tolerance:
                    break
        return ranking_losses


def countAdded(y, y_new):
    percentage_added = [(np.sum(y_new[:,t]) / np.sum(y[:,t])) for t in range(y.shape[1])]
    return np.array(percentage_added)

percentage_added = dict()
for threshold in [0.02, 0.05, 0.2, 0.5, 0.9, 1]:
    percentage_added[threshold] = countAdded(y, y_dict[threshold])

for targ in tqdm(np.arange(y.shape[1])):
    print("Targ:"+str(targ), end=' ')
    for threshold in [0.02, 0.05, 0.2, 0.5, 0.9, 1]:
        if percentage_added[threshold][targ] > percentage_added[1][targ]: #only need to do comparisons if we actually predicted any new ligands onto this label
            #y_new comes first
            vsbs = VirtualScreeningBootstrapper(x, y_dict[threshold], y, BernoulliNB(), numNegs=0.1)
            rloss = vsbs.bootStrap(targ, 0.525, 150, tolerance=2.5, numJobs=18)
            np.save('comparison_full/target_'+str(targ)+'_threshold'+str(threshold)+'.npy', np.array(rloss)) 
            print(threshold, np.around(np.mean(rloss), 3), np.around(np.median(rloss), 3))
            
    for threshold in [1]:
        vsbs = VirtualScreeningBootstrapper(x, y_dict[threshold], y, BernoulliNB(), numNegs=0.1) ##This is equivalent to just the original y. i.e. no added labels.
        rloss = vsbs.bootStrap(targ, 0.525, 150, tolerance=5, numJobs=16)
        np.save('comparison_full/target_'+str(targ)+'_threshold'+str(threshold)+'.npy', np.array(rloss))
        print(threshold, np.around(np.mean(rloss), 3), np.around(np.median(rloss), 3))
        print(' ')

