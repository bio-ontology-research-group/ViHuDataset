# ViHuDataset
Virus-Human dataset for phenotype-based interaction prediction


# Data preparation

- Get the `idmapping.dat.gz` file from UniprotKB

```
gunzip idmapping.dat.gz
grep "Gene_Name" idmapping.dat > uniprot_to_gene_name.tsv
```

# Generate dataset

```
python generate_dataset.py
```
