# Description
This data was synthetically generated for the purpose of testing algorithms that 
detect biclusters. A bicluster is a set of samples where a set of genes is 
expressed different than in all other samples.

## Folders
### A
Easiest to cluster

This folder contains data where the clusters do not overlap.

### B 
Medium difficulty to cluster

overlapping bi clusters on one axis between the clusters

### C
Hard difficulty to cluster

overlapping bi clusters on both axis (genes and samples)

## Files
The filename gives some information about the data already:
- n_genes: The number of genes per cluster(the higher the easier to cluter)
- m: this is the mean used to generate the data
- std: the standard deviation used to generate the data
- overlap: bool telling if there is any overlap between the clusters
- .biclusters/.exprs_z: 
    - .biclusters file contain all information about the clusters:
        - each row represents a cluster
        - column genes are the genes that are differently expressed for that cluster
        - column samples is the samples that are part of this cluster
        - column frac is the percentage of the total samples belonging to this cluster
        - column n_genes is the number of genes of this cluster
        - column n_samples is the number of samples of this cluster
- .exprs_z: the expression data. The data is normalized and can be used directly
            rows are genes, columns are samples
 