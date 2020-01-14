# Datasets

We use Morgan fingerprints with radius 2, of size 256. While this is typically smaller than what one might use in production, anecdotally we and others have found that folding to this size still works surprisingly well. Combined with the additional speed boost, using a smaller fingerprint is ideal for methods development, such as this paper, which uses many repeat computations for robust evaluation. During production (i.e. drug discovery), the block bootstrapping method would not be required and so larger fingerprints can be used.  

* `x_norm_sparse.npz` is a (252409, 256) sparse matrix containing the fingerprint vectors
* `y.npz` is a (252409, 243) sparse matrix containing the label vectors
* `allSmiles.csv` contains the the SMILES codes for the 252,409 ligands, as they appear in ChEMBL24
* `targetNames.csv` contains the common name of the 243 protein targets, as they appear in ChEMBL24
* `allLigands.csv` and `target_IDs.csv` are the chemblIDs of the ligands and targets

