# Part 2.

Includes: 
* Ipython notebook showing how label correlations can be used to fill in missing labels, and how filling in those labels subsequently improves ligand based virtual screening. 

* `run_comparison.py`, which will generate label matrices using multiple threshold values from the correlation graph, and then use the block bootstrapping technique to show how these label matrices lead to improved performance. 

* already calculated data from `run_comparison.py` in `comparison_full` 

* `negatives.csv` which we use to show that 0.01 % (not fraction) of ligands predicted by the label correlation matrix approach are actual known negatives. 
