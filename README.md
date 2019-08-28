# Missing_label_problem
Data and analysis scripts for the paper "The missing label problem: Addressing false assumptions improves ligand-based virtual screening" by Martin LJ and Bowen MT, 2019 (link to chemrxiv goes here)[en.wikipedia.org]


## Layout
Each folder has data or code + explanatory notebooks:
* 'TheData' has the dataset used - molecular fingerprints and label vectors from ChEMBL.
* 'Part_1' demonstrates our block bootstrapping technique that reduces bias during performance evaluation. 
* 'Part_2' contains the code used to fill in missing labels based on correlations between targets.
* 'Part_3' contains the code for unsupervised clustering of ligand sets into their binding sites. 


## Blurb:
This paper explores the effect that filling in missing labels has on ligand-based virtual screening (LBVS). 
LBVS uses featurized molecule instances called molecular fingerprints to fit a mapping function (i.e. machine learning) that can then 
map unseen instances into a set of labels describing the protein targets that the ligand binds to. The matrix made up of
the label vectors is typically very sparse as a result of having large numbers of ligands, each of which only binds to a small number of proteins.

Unfortunately the label matrix is also incomplete, with two types of missing labels. This reduces the ability of 
machine learning models to rank ligands in the correct order, and hasn't been widely recognised or addressed. We use the ChEMBL database as an example.
* a vast majority of protein-ligand interactions in ChEMBL 
have not been tested experimentally yet. Some of those interactions are actually true positives that 
haven't been found yet, as evidenced by the continued 
updates of the ChEMBL database as med chemists and pharmacologists test the interactions and publish results
 in the literature. 
 * binding site labels are unknown. Many proteins have multiple binding sites. From the similar property principle we expect
 all ligands within a binding site to have some common properties. If they bind to different sites, there is no longer any 
 selective pressure on the ligands to have common properties, and so the ligands from each site should be treated separately in the label matrix.
 
 In this paper we took a statistical approach and showed solutions to these two problems _without_ requiring mass in vitro screening 
 to fill in the missing interactions or mass crystallization to determine the binding sites. 
 
 To show the improvements we needed a performance metric that is robust to bias. Many ligands in these databases come from 
 SAR studies, where sometimes one or two atoms are the only difference. As such, the data is no longer i.d.d., and random-split 
 cross-validation is now recognised to be a high-bias
 evaluation technique. We use a technique that is akin to leave-one-out cross validation, but instead we leave out blocks of highly correlated ligands. The blocks are defined by Dice distance to a central ligand, and we determined an appropriate distance cut-off to measure out one 'block' using Gaussian mixture models. 
 Because there are odd and multidimensional distributions of the correlations between ligands, we bootstrap the evaluation until the metric converges. Hence: block bootstrapping.
 
 ## Code
 All of the code used in this paper is available along with notebooks describing each step. The block bootstrapping technique mentioned above 
 can be used like so:
 
 ```

x = #load up 2d numpy array of fingerprints
y = #load up 2d multi-label array of label vectors

vb = VirtualScreeningBootstrapper(x, y_new, BernoulliNB(), numNegs=0.1)

distance_cutoff = 0.525 #dice distance of the nearest neighbours cutoff
number_of_repeats = 150 #how many times to try repeating.
jobs_per_repeat = 16 #how many cores to run on. Don't run a classifier that using multiple cores if you use this
early_stopping_threshold = 2.5 #percentage threshold for convergence of the median ranking loss. 
target_ID = 0 #choose any column number of the multi-label array of label vectors, each column corresponding to a protein target. 

#measure block bootstrapped ranking losses
rlosses = vsbs.bootStrap(targ, 
    distance_cutoff, 
    number_of_repeats, 
    tolerance=early_stopping_threshold, 
    numJobs=jobs_per_repeat)
 
 ```
