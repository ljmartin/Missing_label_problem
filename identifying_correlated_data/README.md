# Identifying correlated data

Molecular bioactivity databases often have some highly similar instances that were derived from a single instance. Because of this, they do not resemble iid data. This holds an IPython notebook for identifying highly correlated, non-iid data in molecular bioactivity databases

This notebook also demonstrates test/train splits using 'blocks' made up of the correlated data, so that you don't have one instance from the test set that was derived from an instance in the train set.  
