# Filling in missing labels

Creating a matrix made up of protein/ligand interaction data leads to a sparse matrix with many zeros. A large majority of these zeros aren't true negatives - they are simply unknown interactions. Most machine learning algorithms see a zero as an explicit negative, leading to incorrect mapping functions. Here, we fill in missing labels using the correlation between similar labels, show that these labels are consistent with the ground truth labels, and then scrape assay data from pubchem showing that many of the predicted labels are actually correct! 

Includes: 
* Ipython notebook showing how label correlations can be used to fill in missing labels, and how filling in those labels subsequently improves ligand based virtual screening. 

* `run_comparison.py`, which will generate label matrices using multiple threshold values from the correlation graph, and then use the block bootstrapping technique to show how these label matrices lead to improved performance. 

* already calculated data from `run_comparison.py` in `comparison_full` 

* `negatives.csv` which we use to show that only 0.01 % of ligands predicted by the label correlation matrix approach are actual known negatives. 

* `finding_puchem.ipynb`, which demonstrates downloading assay data from PubChem to check the validity of the predicted labels
