import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.models import RandomWalkPlusW2VModel
from mowl.utils.random import seed_everything
from mowl.owlapi import OWLAPIAdapter
from mowl.projection import OWL2VecStarProjector, TaxonomyWithRelationsProjector
from mowl.walking import Node2Vec
from mowl.utils.data import FastTensorDataLoader
from evaluators import HPIEvaluator
from dataset import HPIDataset
from utils import print_as_md
import click as ck
import random
import os
import wandb
import logging
import torch as th
import torch.nn as nn
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option('--embedding_dimension', '-dim', default=100, type=int)
@ck.option('--epochs', '-ep', default=30, type=int)
@ck.option('--window_size', '-ws', default=5, type=int, )
@ck.option('--num_walks', '-nw', default=20, type=int)
@ck.option('--walk_length', '-wl', default=20, type=int)
@ck.option('--p', '-p', default=1.0, type=float)
@ck.option('--q', '-q', default=0.5, type=float)
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_nn", "-onn", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(embedding_dimension, epochs, window_size, num_walks,
         walk_length, p, q, wandb_description, no_sweep, only_nn, only_test):

    wandb_logger = wandb.init(entity='ferzcam', project='hpi_analysis', group='owl2vec', name=wandb_description)

    if no_sweep:
        wandb_logger.log({'embedding_dimension': embedding_dimension,
                          'epochs': epochs,
                          'window_size': window_size,
                          'num_walks': num_walks,
                          'walk_length': walk_length,
                          'p': p,
                          'q': q
                          })
    else:
        embedding_dimension = wandb.config.embedding_dimension
        epochs = wandb.config.epochs
        window_size = wandb.config.window_size
        num_walks = wandb.config.num_walks
        walk_length = wandb.config.walk_length
        p = wandb.config.p
        q = wandb.config.q
    
    seed_everything(42)

    root_dir = "../../data/"
    dataset = HPIDataset(root_dir)    
    
    out_dir = "../../models"

    
    model_filepath = os.path.join(out_dir, f"owl2vec_{embedding_dimension}_{epochs}_{window_size}_{num_walks}_{walk_length}_p_{p}_q_{q}.model")
    corpus_filepath = os.path.join(out_dir, f"corpus_owl2vec_{embedding_dimension}_{epochs}_{window_size}_{num_walks}_{walk_length}_p_{p}_q_{q}.txt")

    device = "cpu" # th.device("cuda" if th.cuda.is_available() else "cpu")
    model = RandomWalkPlusW2VModelNN(device, dataset, model_filepath=model_filepath)
    model.set_projector(OWL2VecStarProjector(bidirectional_taxonomy = False, include_literals = True))
    model.set_walker(Node2Vec(num_walks, walk_length, p=p, q=q, workers=10, outfile=corpus_filepath))
    model.set_w2v_model(vector_size=embedding_dimension, workers=16, epochs=epochs, min_count=1, window=window_size)

    model.set_evaluator(HPIEvaluator, device)

    if not only_test and not only_nn:
        model.train(epochs=epochs)
        model.w2v_model.save(model_filepath)
        vectors_size = len(model.w2v_model.wv)
        print(f"Vectors size: {vectors_size} after training w2v model")
        os.remove(corpus_filepath)

    if only_nn:
        model.train_nn()
        model.save_nn(model.nn_model_filepath)
            
        # model.from_pretrained(model_filepath)
        


    model.test()
        
    # model.evaluate()

    micro_metrics, macro_metrics = model.metrics

    
    print("Test macro metrics")
    print_as_md(macro_metrics)
    print("\nTest micro metrics")
    print_as_md(micro_metrics)
    
    micro_metrics = {f"micro_{k}": v for k, v in micro_metrics.items()}
    macro_metrics = {f"macro_{k}": v for k, v in macro_metrics.items()}
    wandb_logger.log({**micro_metrics, **macro_metrics})



class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x

 
class HPINet(nn.Module):
    def __init__(self, human_gene_embedding_layer, virus_taxon_embedding_layer, embed_dim):
        super().__init__()

        self.human_gene_embedding_layer = human_gene_embedding_layer
        self.virus_taxon_embedding_layer = virus_taxon_embedding_layer

        net = []
        input_length = 2*embed_dim
        nodes = [embed_dim]
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, 1))
        net.append(nn.Sigmoid())
        self.hpi_net = nn.Sequential(*net)

    def forward(self, data, *args):
        if data.shape[1] == 2:
            h = data[:, 0]
            t = data[:, 1]
        elif data.shape[1] == 3:
            h = data[:, 0]
            t = data[:, 2]
        else:
            raise ValueError(f"Data shape not consistent: {data.shape}")
        
        head_embs = self.human_gene_embedding_layer(h)
        tail_embs = self.virus_taxon_embedding_layer(t)

        data = th.cat([head_embs, tail_embs], dim=1)
        return self.hpi_net(data)
    
class RandomWalkPlusW2VModelNN(RandomWalkPlusW2VModel):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        self.nn_model_filepath = self.model_filepath + ".nn"
        self.nn_epochs = 100

        # self.evaluator = HPIEvaluator(self.dataset, device=device)
        
    def load_w2v_embeddings_data(self):
        logger.info(f"Initializing neural network")
        classes = self.dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        self.from_pretrained(self.model_filepath)
        # self.w2v_model = self.w2v_model.load(self.model_filepath)

        
        
        w2v_vectors = self.w2v_model.wv
        print(f"Vocabulary size: {len(w2v_vectors)}")
        
        human_gene_embeddings_dict = {}
        virus_taxon_embeddings_dict = {}
        for class_ in classes:
            if "mowl.borg" in class_:
                if class_ in w2v_vectors:
                    human_gene_embeddings_dict[class_] = w2v_vectors[class_]
                else:
                    # logger.warning(f"Class {class_} not found in w2v model")
                    human_gene_embeddings_dict[class_] = np.random.rand(self.embed_dim)
            if "NCBITaxon_" in class_:
                if class_ in w2v_vectors:
                    virus_taxon_embeddings_dict[class_] = w2v_vectors[class_]
                else:
                    # logger.warning(f"Class {class_} not found in w2v model")
                    virus_taxon_embeddings_dict[class_] = np.random.rand(self.embed_dim)
                    
        human_gene_embedding_to_id = {cls: idx for idx, cls in enumerate(human_gene_embeddings_dict.keys())}
        virus_taxon_embedding_to_id = {cls: idx for idx, cls in enumerate(virus_taxon_embeddings_dict.keys())}
        
        human_gene_embeddings_list = np.array(list(human_gene_embeddings_dict.values()))
        virus_taxon_embeddings_list = np.array(list(virus_taxon_embeddings_dict.values()))
        human_gene_embeddings = th.tensor(human_gene_embeddings_list, dtype=th.float).to(self.device)
        virus_embeddings = th.tensor(virus_taxon_embeddings_list, dtype=th.float).to(self.device)
        human_gene_embedding_layer = nn.Embedding.from_pretrained(human_gene_embeddings).to(self.device)
        virus_taxon_embedding_layer = nn.Embedding.from_pretrained(virus_embeddings).to(self.device)
        
        human_gene_embedding_layer.weight.requires_grad = False
        virus_taxon_embedding_layer.weight.requires_grad = False

        hpi_net = HPINet(human_gene_embedding_layer,
                         virus_taxon_embedding_layer, self.embed_dim).to(self.device)
        
        projector = TaxonomyWithRelationsProjector(relations=["http://mowl.borg/associated_with"])

        train_edges = projector.project(self.dataset.ontology)
        valid_edges = projector.project(self.dataset.validation)
        test_edges = projector.project(self.dataset.testing)

        train_indices = [(human_gene_embedding_to_id[e.src], virus_taxon_embedding_to_id[e.dst]) for e in train_edges]
        human_gene_ids = list(human_gene_embedding_to_id.values())
        virus_taxon_ids = list(virus_taxon_embedding_to_id.values())
        
        train_negatives = [(random.choice(human_gene_ids), e[1]) for e in train_indices]
        train_data = train_indices + train_negatives
        labels = [0]*len(train_indices) + [1]*len(train_negatives)
        valid_data = [(human_gene_embedding_to_id[e.src], virus_taxon_embedding_to_id[e.dst]) for e in valid_edges]
        test_data = [(human_gene_embedding_to_id[e.src], virus_taxon_embedding_to_id[e.dst]) for e in test_edges]

        train_data = th.tensor(train_data, dtype=th.long)
        train_labels = th.tensor(labels, dtype=th.long)
        valid_data = th.tensor(valid_data, dtype=th.long)
        test_data = th.tensor(test_data, dtype=th.long)
        
        return hpi_net, train_data, train_labels, valid_data, test_data

    def train_nn(self):
        # super().train()
        # self.w2v_model.save(self.model_filepath)

        hpi_net, train_data, train_labels, valid_data, test_data = self.load_w2v_embeddings_data()

        train_dataloader = FastTensorDataLoader(train_data, train_labels, batch_size=self.batch_size, shuffle=True)

        optimizer = th.optim.AdamW(hpi_net.parameters(), lr = self.learning_rate)

        hpi_net = hpi_net.to(self.device)

        criterion = nn.BCELoss()

        epochs = 1000
        best_mr = float("inf")
        best_mrr = 0
        tolerance = 5
        curr_tolerance = 5
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, batch_labels in train_dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                head = batch_data[:, 0]
                tail = batch_data[:, 1]
                data = th.vstack([head, tail]).T
                
                logits = hpi_net(data).squeeze()
                loss = criterion(logits, batch_labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
                
            

            hpi_net.eval()

            metrics = self._evaluator.evaluate(hpi_net, mode="valid")
            valid_mr = metrics["valid_mr"]
            valid_mrr = metrics["valid_mrr"]

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                th.save(hpi_net.state_dict(), self.nn_model_filepath)
                curr_tolerance = tolerance
                logger.info(f"New best model found: {valid_mr:1f} - {valid_mrr:4f}")
            else:
                curr_tolerance -= 1
                
            logger.info(f"Epoch {epoch}: {epoch_loss} - Valid MR: {valid_mr:1f} - Valid MRR: {valid_mrr:4f}")

            if curr_tolerance == 0:
                logger.info("Early stopping")
                break
            
            
    def test(self):
        self.from_pretrained(self.model_filepath)
        hpi_net, _, _, _, _ = self.load_w2v_embeddings_data()
        hpi_net.to(self.device)
        hpi_net.load_state_dict(th.load(self.nn_model_filepath))
        hpi_net.eval()

        return self._evaluator.evaluate(hpi_net, mode="test")
        
        
        # evaluation_module = EvaluationModel(self.w2v_model, self.dataset, self.embed_dim, self.device)
        
        # return self._evaluator.evaluate(evaluation_module)




class EvaluationModel(nn.Module):
    def __init__(self, w2v_model, dataset, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        
        self.human_gene_embeddings, self.virus_taxon_embeddings = self.init_module(w2v_model, dataset)


    def init_module(self, w2v_model, dataset):
        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        w2v_vectors = w2v_model.wv
        human_gene_embeddings_list = []
        virus_taxon_embeddings_list = []
        for class_ in classes:
            if "mowl.borg" in class_:
                if class_ in w2v_vectors:
                    human_gene_embeddings_list.append(w2v_vectors[class_])
                else:
                    logger.warning(f"Class {class_} not found in w2v model")
                    human_gene_embeddings_list.append(np.random.rand(self.embedding_size))
            if "NCBITaxon_" in class_:
                if class_ in w2v_vectors:
                    virus_taxon_embeddings_list.append(w2v_vectors[class_])
                else:
                    logger.warning(f"Class {class_} not found in w2v model")
                    virus_taxon_embeddings_list.append(np.random.rand(self.embedding_size))

                    
        human_gene_embeddings_list = np.array(human_gene_embeddings_list)
        virus_taxon_embeddings_list = np.array(virus_taxon_embeddings_list)
        human_gene_embeddings = th.tensor(human_gene_embeddings_list).to(self.device)
        virus_embeddings = th.tensor(virus_taxon_embeddings_list).to(self.device)
        return nn.Embedding.from_pretrained(human_gene_embeddings), nn.Embedding.from_pretrained(virus_embeddings)
        
        
    def forward(self, data, *args, **kwargs):

        x = data[:, 0]
        y = data[:, 1]

        logger.debug(f"X shape: {x.shape}")
        logger.debug(f"Y shape: {y.shape}")
        
        x = self.human_gene_embeddings(x)
        y = self.virus_taxon_embeddings(y)

        logger.debug(f"X shape: {x.shape}")
        logger.debug(f"Y shape: {y.shape}")
        
        dot_product = th.sum(x * y, dim=1)
        logger.debug(f"Dot product shape: {dot_product.shape}")
        return 1 - th.sigmoid(dot_product)


    
if __name__ == "__main__":
    main()
