from mowl.datasets import PathDataset, OWLClasses
import os


class HPIDataset(PathDataset):

    def __init__(self, root_dir):
        root = os.path.abspath(root_dir)
        train_path = os.path.join(root, "train.owl")
        valid_path = os.path.join(root, "valid.owl")
        test_path = os.path.join(root, "test.owl")

        super(HPIDataset, self).__init__(train_path, valid_path, test_path)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            viruses = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "NCBITaxon_" in owl_name:
                    viruses.add(owl_cls)

            genes = OWLClasses(genes)
            viruses = OWLClasses(viruses)
            self._evaluation_classes = (genes, viruses)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

    
