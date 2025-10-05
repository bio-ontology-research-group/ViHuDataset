# ViHuDataset
Virus-Human dataset for phenotype-based interaction prediction


# Data

Most data files are under `data/`. You might need to uncompress `phenomenet-inferred.owl.gz` and `train.owl.gz`

# Running experiments

```
cd src/
```

- Semantic similarity

```
groovy semantic_similarity.groovy
```


- OWL2Vec*

```
cd src/owl2vec
python hpi.py --ns
```

- OWL2Vec* projection + TransE

```
cd src/owl2vec_kge
python hpi.py --ns
```


# Generate dataset

Data is provided in the repostory. In the case you want to regenerate the data, please run:

```
python generate_dataset.py
```

