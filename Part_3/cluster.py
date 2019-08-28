import numpy as np
from tqdm import tqdm
import itertools
from numba import jit
import pandas as pd
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score
from scipy import sparse
from scipy import stats
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from joblib import Parallel, delayed

import copy

x = sparse.load_npz('../TheData/x_norm_sparse.npz')
y = sparse.load_npz('../TheData/y.npz')

x = np.array(x.todense())
y = np.array(y.todense())
allSmiles = pd.read_csv('../TheData/allSmiles.csv', header=None)
targetNames = pd.read_csv('../TheData/targetNames.csv', header=None)

##Creating the x values for clustering - we prefer higher dimensionality here to better distinguish binding sites
mols = [Chem.MolFromSmiles(i) for i in allSmiles[0]]
fpsMorgan = [AllChem.GetMorganFingerprintAsBitVect(m2,2,2048) for m2 in tqdm(mols)]
x_2_2048 = np.array([[int(i) for i in fp.ToBitString()] for fp in tqdm(fpsMorgan)])


from scipy.spatial import distance
same_label = list()
different_label = list()

##Best separation comes from Dice:
vec_distance = lambda a,b: distance.dice(a,b)
sample_dist = lambda a: np.random.choice(a, 500, replace=False)

for _ in range(15):
    j, k = np.random.choice(y.shape[1], 2, replace=False)
    
    j_instances = x_2_2048[y[:,j]==1]
    k_instances = x_2_2048[y[:,k]==1]
    
    for rep in range(500):
        j_inst = j_instances[np.random.choice(j_instances.shape[0], 1)]
        k_inst = k_instances[np.random.choice(k_instances.shape[0], 1)]
        different_label.append(vec_distance(j_inst, k_inst))
        
    for rep in range(250):
        j1_inst, j2_inst = np.random.choice(j_instances.shape[0], 2, replace=False)
        same_label.append(vec_distance(j_instances[j1_inst], j_instances[j2_inst]))
        
    for rep in range(250):
        k1_inst, k2_inst = np.random.choice(k_instances.shape[0], 2, replace=False)
        same_label.append(vec_distance(k_instances[k1_inst], k_instances[k2_inst]))
        
def getPairPredictions(clusters, distance_matrix, sames, differents, verbose=0):
    pairList = dict()
    pList = list()
    if verbose:
        print('Calculating pairwise probabilities for %s clusters' % len(clusters))

    count=0
    for p1,p2 in itertools.combinations(range(len(clusters)), 2):
        print(p1,' ', p2, end='\r')
        indices = np.concatenate([clusters[p1], clusters[p2]])
        joined_matrix = copy.copy(distance_matrix[indices][:,indices])      
        flattened_joined = joined_matrix[joined_matrix>0]
        
        joined_ks_same = stats.ks_2samp(sames,  flattened_joined)[0]
        joined_ks_diff = stats.ks_2samp(differents,  flattened_joined)[0]
        
        pList.append(joined_ks_diff/joined_ks_same)
        
        pairList[count] = [p1,p2]
        count+=1
    biggest = np.argmax(pList)
    print(biggest, pList[biggest])
    if pList[biggest]>1:
        maxPair = pairList[biggest]
        p1,p2=maxPair
        clusters[p1] = np.concatenate([clusters[p1], clusters[p2]])
        mask = np.ones(len(clusters), bool)
        mask[p2]=False
        clusters = np.array(clusters)[mask]
        return list(clusters)
    else: 
        return clusters
    
sample_dist = lambda a: np.random.choice(a, 500, replace=False)
same_sample = sample_dist(same_label)
different_sample = sample_dist(different_label)


def getClusters(target_index, x_in, y_in, verbose=0):
    x_positive = np.array(x_in[y_in[:,target_index]==1], dtype=bool) #pairwise_distances likes boolean
    distance_matrix = pairwise_distances(x_positive, metric='dice')
    clusterer = AgglomerativeClustering(n_clusters=None,
                                     affinity='precomputed', 
                                     linkage='average',
                                    distance_threshold=0.8)

    clusterer.fit(distance_matrix)
    numclust = clusterer.labels_.max()+1
    clusters = [np.where(clusterer.labels_==i)[0] for i in range(numclust)]
    while len(clusters)>1:
        oldnum = copy.copy(numclust)
        clusters = getPairPredictions(clusters, 
                                      distance_matrix, 
                                      same_label,
                                      different_label,
                                     verbose=verbose)
        numclust= len(clusters)
        if verbose:
            print('\tNumber of clusters:', len(clusters))
        if numclust==oldnum:
            break

    #if cluster has too few members to do meaningful analysis, fold them into their kNN. 
    cluster_labels = np.zeros(len(x_positive))
    for count, c in enumerate(clusters):
        cluster_labels[c]=count

    lengths = [len(c) for c in clusters]

    while min(lengths)<50:
        shortest = np.argmin(lengths)
        for ligand in clusters[shortest]:
            sorted_distances = np.argsort(distance_matrix[ligand])
            nearest_labels = cluster_labels[sorted_distances]
    
            for count, label in enumerate(nearest_labels):
                if label!=shortest:
                    cluster_labels[ligand]=label
                    break
        clusters = [np.where(cluster_labels==i)[0] for i in np.unique(cluster_labels)]
        cluster_labels = np.zeros(len(x_positive))
        for count, c in enumerate(clusters):
            cluster_labels[c]=count

        lengths = [len(c) for c in clusters]

    if verbose:
        print('DONE, number of clusters is %s' % len(clusters))
    return clusters


target_id_map = dict()
new_target_id = 0
y_new = copy.copy(y)

y_new = np.zeros([y.shape[0],0]) #this is a blank column vector
for targ in range(y.shape[1]):
    target_id_map[targ] = list()
#    x_positive = x_2_2048[y[:,targ]==1] #Grab the high-D x vectors for this target
    positive_indices = np.where(y[:,targ]==1)[0] #Grab the indices for those vectors
    print(targetNames.iloc[targ]+' - ', end=' ')
    
        #Do the clustering:
    clusters = getClusters(targ, x_2_2048, y)
    print('Clusters %s' % len(clusters))
        #iterate through clusters, making a label column vector for each (even if it's a single cluster)

    for cluster in clusters:
        labels = np.zeros([y.shape[0], 1])
        labels[positive_indices[cluster]]=1
        y_new = np.hstack([y_new, labels])
        target_id_map[targ].append(new_target_id)
        new_target_id+=1

np.save('y_cluster.npy', y_new)
f = open('y_cluster_map.dat', 'w')
for count in range(243):
    f.write(str(count)+': ')
    for j in target_id_map[count]:
        f.write(str(j)+' ')
        
    f.write('\n')
    
f.close()


class VirtualScreeningBootstrapper():
    def __init__(self, x, y, clf, numNegs=0.2):
        self.x = x
        self.y = y
        self.clf = clf

        self.positive_indices = None
        self.negative_indices = None
        self.dist = None
        self.targetIndex = None
        self.fraction = None
        self.numNegs = numNegs

    def setTarget(self, target_index):
        self.targetIndex = target_index
        self.positive_indices = np.where(self.y[:,self.targetIndex]==1)[0]
        self.negative_indices = np.where(self.y[:,self.targetIndex]==0)[0]
        self.weights = np.ones(len(self.positive_indices))

        print(len(self.positive_indices), len(np.where(self.y[:,self.targetIndex]==1)[0]))
        self.dist = pairwise_distances(np.array(self.x[self.positive_indices], dtype=bool), metric='dice')

    def maskNN(self):
        self.weights=normalize([self.weights], norm='l1')[0]
        selection = np.random.choice(self.positive_indices, p=self.weights) #take a random true positive ligand
        ligand_index = np.where(self.positive_indices==selection)[0][0]

        knn = np.where(self.dist[ligand_index]<self.fraction)[0] #these are the nearest neighbours
        add=0.01
        while len(knn)<10: #ensure we have at least 10 positive ligands.
            knn = np.where(self.dist[ligand_index]<(self.fraction+add))[0]
            add+=0.01

        distances = self.dist[ligand_index][knn]
        self.weights[knn] *= (distances+0.01) #change the weights

        mask = np.ones(len(self.x), dtype=bool) #create a mask
        mask[self.positive_indices[knn]]=False #set only the neighbors to False
        return mask

    def maskNegs(self, mask):
        neg_selection = np.random.choice(self.negative_indices, int(self.numNegs*len(self.negative_indices)))
        mask[neg_selection]=False
        return mask

    def evaluateFold(self, clf, mask):
        probs = clf.predict_proba(self.x[~mask])[:,1] #get probability of test ligands being positive
        ranking_loss = label_ranking_loss(self.y[~mask][:,self.targetIndex].reshape(1,-1), probs.reshape(1,-1))
        return ranking_loss

    def bootstrap_par(self, mask):
        clf = copy.copy(self.clf)
        clf.fit(self.x[mask], self.y[mask][:,self.targetIndex])
        ranking_loss = self.evaluateFold(clf, mask)
        return ranking_loss

    def makeMask(self):
        mask = self.maskNN() #generate a mask for a random block of positives
        mask = self.maskNegs(mask) #mask some negatives too
        return mask

    def bootStrap(self, target_index, fraction, repeats, tolerance=2.5, numJobs=1):
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
                mask = self.makeMask()
                ranking_loss = self.bootstrap_par(mask)
                ranking_losses.append(ranking_loss)

            if len(ranking_losses)>53:
                medians = [np.median(ranking_losses[:-i]) for i in np.arange(1,51)]
                diffs = [np.abs(medians[-1]- median) for median in medians[:-1]]
                diffs_percent = [(d / medians[-1])*100 for d in diffs]
                if max(diffs_percent)<tolerance:
                    break
        return ranking_losses

print('clustering is done. Now calculating RLoss of each cluster for proteins with multiple sites')

vsbs = VirtualScreeningBootstrapper(x, y_new, BernoulliNB(), numNegs=0.1)

for j,k in target_id_map.items():
    if len(k)>1:
        for targ in k:
            print("Original targ is: %s and binding site number is: %s" % (str(j), str(targ)))
            print("There are %s ligands in this site" % len(k))
            rloss = vsbs.bootStrap(targ, 0.525, 150, tolerance=2.5, numJobs=24)
            np.save('cluster_full/target_'+str(targ)+'.npy', np.array(rloss))
