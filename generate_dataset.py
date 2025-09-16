import mowl
mowl.init_jvm("10g")
import click as ck
import pandas as pd
import logging
from jpype import *
import jpype.imports
import os
import wget
import sys
import json
from tqdm import tqdm
import random


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import OWLClasses
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model import IRI
import java
from java.util import HashSet

adapter = OWLAPIAdapter()
manager = adapter.owl_manager
factory = adapter.data_factory

seed_everything(42)


@ck.command()
@ck.option(
    '--save_dir', '-s', default='data', help='Directory to save the data')
def main(save_dir):

    out_dir = os.path.abspath(save_dir)
    logger.info(f'Saving data to {out_dir}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Cheking if the data is already downloaded")

    if not os.path.exists(os.path.join(out_dir, 'phenomenet-inferred.owl')):
        logger.error("File phenomenet-inferred.owl not found.")
        
    if not os.path.exists(os.path.join(out_dir, 'MGI_GenePheno.rpt')):
        logger.error("File MGI_GenePheno.rpt not found. Downloading it for Gene-Phenotype associations")
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt", out=out_dir)

    if not os.path.exists(os.path.join(out_dir, 'HMD_HumanPhenotype.rpt')):
        logger.error("File HMD_HumanPhenotype.rpt not found. Downloading it for MGI to Entrez Gene mappings")
        wget.download("https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt", out=out_dir)

    if not os.path.exists(os.path.join(out_dir, 'pathogens.4web.json')):
        logger.error("File pathogens.4web.json not found. Required for Pathogen-Disease associations")
        
    if not os.path.exists(os.path.join(out_dir, 'hpidb2.mitab.txt')):
        logger.error("File hpidb2.mitab.txt not found. Required for Virus-Host interactions")
        
    logger.info("Loading ontology")
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(os.path.join(out_dir, 'phenomenet-inferred.owl')))
    classes = set(OWLClasses(ont.getClassesInSignature()).as_str)

    existing_mp_phenotypes = set()   # 
    existing_hp_phenotypes = set()
    for cls in classes:
        if "MP_" in cls:
            existing_mp_phenotypes.add(cls)
        elif "HP_" in cls:
            existing_hp_phenotypes.add(cls)
    logger.info(f"Existing MP phenotypes in ontology: {len(existing_mp_phenotypes)}")
    logger.info(f"Existing HP phenotypes in ontology: {len(existing_hp_phenotypes)}")

    virus_has_phenotype = "http://mowl.borg/virus_has_phenotype"
    gene_has_phenotype = "http://mowl.borg/gene_has_phenotype"
    associated_with = "http://mowl.borg/associated_with"


    logger.info("Obtaining MGI to Entrez Gene mappings")
    hmd = pd.read_csv(os.path.join(out_dir, 'HMD_HumanPhenotype.rpt'), sep='\t')
    hmd.columns = ["HGene", "EntrezID", "MGene", "MGI ID", "Phenotypes", "Extra"]

    hgene_to_entrez = dict()
    for i, row in hmd.iterrows():
        hgene_to_entrez[row["HGene"]] = row["EntrezID"]
    
    mgi_to_entrez = dict()
    for i, row in hmd.iterrows():
        mgi_to_entrez[row["MGI ID"]] = row["EntrezID"]

    
    logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")

    mgi_gene_pheno = pd.read_csv(os.path.join(out_dir, 'MGI_GenePheno.rpt'), sep='\t', header=None)
    mgi_gene_pheno.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "empty", "MGI Genotype ID"]

    gene_phenotypes = []
    for index, row in mgi_gene_pheno.iterrows():
        genes = row["MGI ID"]
        phenotype = row["MP Phenotype"]
        assert phenotype.startswith("MP:")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        if not phenotype in existing_mp_phenotypes:
            continue

        for gene in genes.split('|'):
            if gene in mgi_to_entrez:
                gene = mgi_to_entrez[gene]
                gene = "http://mowl.borg/" + str(gene) #.replace(":", "_")
                gene_phenotypes.append((gene, phenotype))

    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"\tE.g. {gene_phenotypes[0]}")
    genes_with_phenotypes = set([gene for gene, _ in gene_phenotypes])
    logger.info(f"Genes with phenotypes: {len(genes_with_phenotypes)}")
    
    
    logger.info("Obtaining Virus-Phenotype associations from pathogens.4web.json. Viruses are represented as NCBI Taxon IDs and Phenotypes are represented as HP IDs")

    if not os.path.exists(os.path.join(out_dir, 'virus_taxids.tsv')):
        logger.error("File virus_taxids.tsv not found. Required for Virus TaxIDs. Run 'python get_virus_taxids.py' to generate it.")
    virus_taxids_df = pd.read_csv(os.path.join(out_dir, 'virus_taxids.tsv'), sep='\t')
    viral_taxids = virus_taxids_df['taxid'].astype(str).tolist()
    viral_taxids = ["http://purl.obolibrary.org/obo/NCBITaxon_" + taxid for taxid in viral_taxids]
    logger.info(f"Total viral taxids: {len(viral_taxids)}")
    logger.info(f"\tE.g. {viral_taxids[:5]}")
    
    
    with open(os.path.join(out_dir, 'pathogens.4web.json')) as f:
        taxa_data = json.load(f)

    logger.info(f"Total taxa in pathogens.4web.json: {len(taxa_data)}")

    virus_phenotypes = []
    for entry in tqdm(taxa_data, desc="Processing taxa"):
        taxid_uri = entry["TaxID"]

        # Check if this taxon is a virus
        if not taxid_uri in viral_taxids:
            continue

        phenotypes = entry["Phenotypes"]

        for phenotype in phenotypes:
            if not phenotype.startswith("http://purl.obolibrary.org/obo/HP_"):
                continue
            if phenotype not in existing_hp_phenotypes:
                continue

            virus_phenotypes.append((taxid_uri, phenotype))

    logger.info(f"Virus-Phenotype associations: {len(virus_phenotypes)}")
    logger.info(f"\tE.g. {virus_phenotypes[0]}")
    viruses_with_phenotypes = set([virus for virus, _ in virus_phenotypes])
    logger.info(f"Viruses with phenotypes: {len(viruses_with_phenotypes)}")
    
    logger.info("Obtaining Virus-Human associations from HPIDB")
    virus_human_associations_df = pd.read_csv(os.path.join(out_dir, 'virus_human_associations.tsv'), sep='\t')
    virus_human_associations = []

    skipped_by_virus = 0
    skipped_by_human = 0
    skipped_by_nan = 0
    skipped_by_non_existing_human_gene = 0
    for i, row in virus_human_associations_df.iterrows():
        virus_taxa = "http://purl.obolibrary.org/obo/NCBITaxon_" + str(row['virus_taxa'])
        if not virus_taxa in viruses_with_phenotypes:
            skipped_by_virus += 1
            continue

        if pd.isna(row['human_gene_name']):
            skipped_by_nan += 1
            continue
        human_gene = str(row['human_gene_name'])

        if not human_gene in hgene_to_entrez:
            skipped_by_non_existing_human_gene += 1
            continue
        human_entrez = hgene_to_entrez[human_gene]
        human_entrez = "http://mowl.borg/" + str(human_entrez)  # Assuming human_protein is Entrez Gene ID
        if not human_entrez in genes_with_phenotypes:
            skipped_by_human += 1
            continue
        virus_human_associations.append((virus_taxa, human_entrez))
        
    logger.info(f"Total Virus-Human associations from HPIDB: {len(virus_human_associations)}")
    virus_human_associations = list(set(virus_human_associations))
    logger.info(f"Unique Virus-Human associations after removing duplicates: {len(virus_human_associations)}")
    logger.info(f"\tE.g. {virus_human_associations[0]}")
    logger.info(f"Skipped Virus-Human associations due to missing virus phenotypes: {skipped_by_virus}")
    logger.info(f"Skipped Virus-Human associations due to missing human gene phenotypes: {skipped_by_human}")
    logger.info(f"Skipped Virus-Human associations due to missing human gene Entrez ID: {skipped_by_nan}")
    logger.info(f"Skipped Virus-Human associations due to human gene not in MGI: {skipped_by_non_existing_human_gene}")
    
    gene_phenotypes = [(gene, phenotype) for gene, phenotype in gene_phenotypes if phenotype in classes]
    virus_phenotypes = [(virus, phenotype) for virus, phenotype in virus_phenotypes if phenotype in classes]
    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"Virus-Phenotype associations: {len(virus_phenotypes)}")
    
        
    gene_phenotype_axioms = HashSet([create_axiom(gene, gene_has_phenotype, phenotype) for gene, phenotype in gene_phenotypes])
    virus_phenotype_axioms = HashSet([create_axiom(virus, virus_has_phenotype, phenotype) for virus, phenotype in virus_phenotypes])

    with open(os.path.join(out_dir, 'gene_to_phenotype.csv'), 'w') as f:
        for gene, phenotype in gene_phenotypes:
            f.write(f"{gene},{phenotype}\n")
    with open(os.path.join(out_dir, 'virus_to_phenotype.csv'), 'w') as f:
        for virus, phenotype in virus_phenotypes:
            f.write(f"{virus},{phenotype}\n")

    with open(os.path.join(out_dir, 'virus_human_associations.csv'), 'w') as f:
        for virus, human in virus_human_associations:
            f.write(f"{virus},{human}\n")

    train, valid, test = split_pairs(virus_human_associations, "7,1,2", split_by="pair")
    print(f"Train size: {len(train)}")
    print(f"Valid size: {len(valid)}")
    print(f"Test size: {len(test)}")
    
    train_axioms = HashSet([create_axiom(gene, associated_with, virus) for virus, gene in train])
    valid_axioms = HashSet([create_axiom(gene, associated_with, virus) for virus, gene in valid])
    test_axioms = HashSet([create_axiom(gene, associated_with, virus) for virus, gene in test])
    
    manager.addAxioms(ont, gene_phenotype_axioms)
    manager.addAxioms(ont, virus_phenotype_axioms)
    manager.addAxioms(ont, train_axioms)

    valid_ontology = manager.createOntology()
    manager.addAxioms(valid_ontology, valid_axioms)

    test_ontology = manager.createOntology()
    manager.addAxioms(test_ontology, test_axioms)
    
                    
    manager.saveOntology(ont, IRI.create('file:' + os.path.abspath("train.owl")))
    manager.saveOntology(valid_ontology, IRI.create('file:' + os.path.abspath("valid.owl")))
    manager.saveOntology(test_ontology, IRI.create('file:' + os.path.abspath("test.owl")))
        
    logger.info("Done")
        

def create_axiom(subclass, property_, filler):
    subclass = adapter.create_class(subclass)
    property_ = adapter.create_object_property(property_)
    filler = adapter.create_class(filler)

    existential_restriction = adapter.create_object_some_values_from(property_, filler)
    subclass_axiom = adapter.create_subclass_of(subclass, existential_restriction)
    return subclass_axiom

def split_pairs(pairs, split, split_by="pair"):
    logger.info(f"Splitting {len(pairs)} pairs into training, validation and test sets with split {split}")

    train_ratio, valid_ratio, test_ratio = [int(x) for x in split.split(",")]
    assert train_ratio + valid_ratio + test_ratio == 10

    if split_by == "pair":
        random.shuffle(pairs)
        num_train = int(len(pairs) * train_ratio / 10)
        num_valid = int(len(pairs) * valid_ratio / 10)
        num_test = len(pairs) - num_train - num_valid

        train = pairs[:num_train]
        valid = pairs[num_train:num_train+num_valid]
        test = pairs[num_train+num_valid:]

        logger.info(f"Train: {len(train)}. Ratio: {len(train)/len(pairs)}")
        logger.info(f"Valid: {len(valid)}. Ratio: {len(valid)/len(pairs)}")
        logger.info(f"Test: {len(test)}. Ratio: {len(test)/len(pairs)}")
        
        #make sure second element of each pair in valid and test exists in train
        train_tails =set([tail for head, tail in train])
        valid_tails = set([tail for head, tail in valid])
        test_tails = set([tail for head, tail in test])

        unseen_in_valid = valid_tails - train_tails
        unseen_in_test = test_tails - train_tails
        logger.info(f"Unseen tails in valid: {len(unseen_in_valid)}")
        logger.info(f"Unseen tails in test: {len(unseen_in_test)}")

        if len(unseen_in_valid) > 0 or len(unseen_in_test) > 0:
            logger.info("Moving unseen tails from valid and test to train")
            new_train = []
            for head, tail in valid + test:
                if tail in unseen_in_valid or tail in unseen_in_test:
                    new_train.append((head, tail))
            train += new_train
            valid = [pair for pair in valid if pair not in new_train]
            test = [pair for pair in test if pair not in new_train]
            logger.info(f"New train size: {len(train)}")
            logger.info(f"New valid size: {len(valid)}")
            logger.info(f"New test size: {len(test)}")

            
        
        assert len(set(train) & set(valid)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(valid) & set(test)) == 0
        
        return train, valid, test

    if split_by == "head":
        raise NotImplementedError

    if split_by == "tail":
        tail_to_heads = dict()
        for head, tail in pairs:
            if tail not in tail_to_heads:
                tail_to_heads[tail] = []
            tail_to_heads[tail].append(head)

        tails = list(tail_to_heads.keys())
        random.shuffle(tails)

        num_train = int(len(tails) * train_ratio / 10)
        num_valid = int(len(tails) * valid_ratio / 10)
        num_test = int(len(tails) * test_ratio / 10)
        
        train_tails = tails[:num_train]
        valid_tails = tails[num_train:num_train+num_valid]
        test_tails = tails[num_train+num_valid:]

        assert len(set(train_tails) & set(valid_tails)) == 0
        assert len(set(train_tails) & set(test_tails)) == 0
        assert len(set(valid_tails) & set(test_tails)) == 0

        train = [(head, tail) for tail in train_tails for head in tail_to_heads[tail]]
        valid = [(head, tail) for tail in valid_tails for head in tail_to_heads[tail]]
        test = [(head, tail) for tail in test_tails for head in tail_to_heads[tail]]

        assert len(set(train) & set(valid)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(valid) & set(test)) == 0

        assert len(train) + len(valid) + len(test) == len(pairs)
        logger.info(f"Train: {len(train)}. Ratio: {len(train)/len(pairs)}")
        logger.info(f"Valid: {len(valid)}. Ratio: {len(valid)/len(pairs)}")
        logger.info(f"Test: {len(test)}. Ratio: {len(test)/len(pairs)}")
        
        return train, valid, test
                                                                                                                                                

if __name__ == '__main__':
    main()
    shutdownJVM()
