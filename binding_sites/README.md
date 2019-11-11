# Binding sites

Many interaction databases lack information about where a ligand binds. If there are multiple binding sites, the similar property principle no longer applies, since there is no expectation that different binding sites on a protein should have the same shape. The attached files demonstrate how to use clustering and merging to identify ligand sets coming from multiple binding sites, and then group those ligand sets into their binding sites. This shows improved lbvs. 

Includes:
* Ipython notebook demonstrating analysis of the pre-computed results, as well as a demo of clustering a GABA-A Receptor ligand set with 4 binding sites.

* `cluster.py` which will run through the dataset and cluster+merge all target ligand sets

* `cluster_full` with the data from `cluster.py`
