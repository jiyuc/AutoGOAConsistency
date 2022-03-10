import torch
import glob
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd
from corpus_path import evidence_passages
from corpus_path import go_info, gene_info, go_hier
from corpus_path import train_csv, dev_csv, test_csv
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dgl
from knowledge_graph import GOKGModel



class MakeGODAG:
    GO_HIER_MAP = {'is_a': 0,
                   'parent_is_a': 1,
                   'part_of': 2,
                   'parent_part_of': 3,
                   'co_term': 4}

    def __init__(self, go_hier_csv):
        print("build GO DAG")

        go_hier_df = pd.read_csv(go_hier_csv)
        go_src = go_hier_df['src'].tolist()
        go_des = go_hier_df['des'].tolist()
        left_go = pd.read_csv(train_csv)['des'].tolist() \
                  + pd.read_csv(dev_csv)['des'].tolist() \
                  + pd.read_csv(test_csv)['des'].tolist()

        # encode nodes to unique node id
        self.le = LabelEncoder().fit(go_src + go_des + left_go)

        # directional go hierarchy
        src = self.le.transform(go_src)
        des = self.le.transform(go_des)
        self.g = dgl.graph((src, des))
        go_hier_elabels = [self.GO_HIER_MAP[l] for l in go_hier_df['edge_type'].tolist()]

        # splitting dataset
        train_mask = torch.zeros(len(go_hier_elabels), dtype=torch.bool).bernoulli(0.7)  # 65%
        for idx in range(len(go_hier_elabels)):
            if go_hier_elabels[idx] == 4:
                train_mask[idx] = True

        test_mask = ~train_mask
        for idx in range(len(go_hier_elabels)):
            if go_hier_elabels[idx] == 4:
                test_mask[idx] = True

        self.g.edata['label'] = torch.tensor(go_hier_elabels, dtype=torch.long)  # edge label
        self.g.edata['train_mask'] = train_mask
        self.g.edata['test_mask'] = test_mask
        #self.g.ndata['feature'] = ExtractNodeFeature(self.g, self.le, max_length=max_length).forward()

    def forward(self, max_length=350):
        self.g.ndata['feature'] = ExtractNodeFeature(self.g, self.le, max_length=max_length).forward()
        return self.g

    def get_node_encoder(self):
        return self.le


class MakeGraph:
    GO_HIER_MAP = {'is_a': 5,
                   'parent_is_a': 6,
                   'part_of': 7,
                   'parent_part_of': 8}

    def __init__(self, goa_csv, go_hier_csv, train=True):
        print("build goa network")
        goa_df = pd.read_csv(goa_csv)
        go_hier_df = pd.read_csv(go_hier_csv)
        goa_src = goa_df['src'].tolist()
        goa_des = goa_df['des'].tolist()

        go_src = go_hier_df['src'].tolist()
        go_des = go_hier_df['des'].tolist()

        # encode nodes to unique node id
        self.le = LabelEncoder().fit(goa_src + goa_des + go_src + go_des)

        # undirectional goa edges
        src = self.le.transform(goa_src)
        des = self.le.transform(goa_des)
        u = np.concatenate([src, des])
        v = np.concatenate([des, src])
        self.g = dgl.DGLGraph((u, v))
        goa_elabels = [l - 1 for l in goa_df['edge_type'].tolist()] * 2
        # if train:
        # train_mask = torch.zeros(len(goa_elabels), dtype=torch.bool).bernoulli(0.4)
        # test_mask = ~train_mask

        # directional go hierarchy
        src = self.le.transform(go_src)
        des = self.le.transform(go_des)
        self.g.add_edges(src, des)
        go_hier_elabels = [self.GO_HIER_MAP[l] for l in go_hier_df['edge_type'].tolist()]

        if train:
            train_mask = torch.zeros(len(goa_elabels), dtype=torch.bool).bernoulli(0.4)
            test_mask = ~train_mask
            train_mask = torch.cat((train_mask, torch.zeros(len(go_hier_elabels), dtype=torch.bool)), 0)
            test_mask = torch.cat((test_mask, torch.ones(len(go_hier_elabels), dtype=torch.bool)), 0)

        else:
            train_mask = test_mask = \
                torch.cat((torch.ones(len(goa_elabels), dtype=torch.bool),
                           torch.zeros(len(go_hier_elabels), dtype=torch.bool)), 1)

        self.g.edata['label'] = torch.tensor(goa_elabels + go_hier_elabels, dtype=torch.long)  # edge label
        self.g.edata['train_mask'] = train_mask
        self.g.edata['test_mask'] = test_mask
        self.g.ndata['feature'] = ExtractNodeFeature(self.g, self.le).forward()

    def forward(self):
        return self.g

    def get_node_encoder(self):
        return self.le


class ExtractNodeFeature:
    def __init__(self, g, le, max_length=350):
        nodes = le.inverse_transform(g.nodes())
        print("extracting node text attribute")

        # go term & definition -> go_dict[go id] = (go term, go definition)
        go_df = pd.read_csv(go_info).drop_duplicates()
        go_dict = go_df.set_index('go_id').T.to_dict('list')

        # gene symbol and gene description -> gene_dict[gene_id] = (gene symbol, gene description)
        gene_df = pd.read_csv(gene_info).drop_duplicates()
        gene_dict = gene_df.set_index('gene_id').T.to_dict('list')

        # evidence info -> textpr[pmid] = (title, abstract)
        textpr = ParsePassage()

        # encode either PMIDgene info pair or go info pair with BERT tokenizer and encoder
        node_features = []
        for n in nodes:
            template = n.split('\t')
            if len(template) == 2:  # PMIDgene node
                pmid = template[0]
                gene_id = int(template[1])

                # flat cat of title and abstract
                evidence = str(textpr.get_title(pmid)) + str(textpr.get_abstract(pmid))

                # flat cat of gene symbol and gene description
                gene = str(gene_dict[int(gene_id)][0]) + str(gene_dict[int(gene_id)][1])

                # encode
                node_features.append((evidence, gene))

            else:  # GO node
                try:
                    go_term = go_dict[n][0]
                    go_def = go_dict[n][1]
                except:
                    print(n)
                    continue

                # encode
                node_features.append((go_term, go_def))
                #node_features.append((go_term))

        # initial input textural features
        self.node_encodings = torch.tensor(Encode(node_features, max_length=max_length).forward()["input_ids"],
                                           dtype=torch.float)
        print('node feature shape:', self.node_encodings.shape)
        # self.g.ndata['feature'] = node_encodings

    def forward(self):
        return self.node_encodings


class Integration:
    def __init__(self, goa_csv):
        print("integrating dataset")
        goa_df = pd.read_csv(goa_csv)
        go_df = pd.read_csv(go_info).drop_duplicates()
        gene_df = pd.read_csv(gene_info).drop_duplicates()
        go_dict = go_df.set_index('go_id').T.to_dict('list')
        gene_dict = gene_df.set_index('gene_id').T.to_dict('list')

        textpr = ParsePassage()
        pmids = [x.split('\t')[0] for x in goa_df['src'].tolist()]
        genes = [x.split('\t')[1] for x in goa_df['src'].tolist()]

        # evidence info
        print("integrating evidence into")
        self.titles = [textpr.get_title(pmid) for pmid in pmids]
        self.abstracts = [textpr.get_abstract(pmid) for pmid in pmids]
        self.gene_symbols = [gene_dict[int(gene_id)][0] for gene_id in genes]
        self.gene_descriptions = [gene_dict[int(gene_id)][1] for gene_id in genes]

        # go info
        print("integrating gene ontology info")
        gos = goa_df['des'].tolist()
        self.go_terms = [go_dict[go_id][0] for go_id in gos]
        self.go_defs = [go_dict[go_id][1] for go_id in gos]

    def forward(self):
        return list(
            zip(self.titles, self.abstracts, self.gene_symbols, self.gene_descriptions, self.go_terms, self.go_defs))


class Preprocessing:
    def __init__(self, title, abstract, gene_symbol, gene_description, go_term, go_def):
        self.title = str(title)
        self.abstract = str(abstract)
        self.gene_symbol = str(gene_symbol)
        self.gene_description = '[MASK]'#str(gene_description)
        self.go_term = str(go_term)
        self.go_def = '[MASK]'#str(go_def)

    def forward(self):
        """
        extract text attribute for each GOA annotation
        :return: tuple of (gene_conditioned_evidence, go_info)
        """
        if not self.abstract:
            self.abstract = ''

        # flat concatenation at text level
        evidence = self.title + ' ' + self.abstract
        gene_info = self.gene_symbol + ' ' + self.gene_description
        gene_conditioned_evidence = evidence + ' ' + gene_info
        go_info = self.go_term + ' ' + self.go_def
        return gene_conditioned_evidence, go_info


class DatasetBert(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, go_hier_encodings, binary=False):
        print("converting to BERT dataset format")
        self.encodings = encodings
        self.go_hier_encodings = go_hier_encodings
        if not binary:
            self.labels = labels
        elif binary:
            self.labels = []
            for label in labels:
                if label == 0:
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __getitem__(self, idx):
        # pre joint
        item = {key: torch.tensor(val[idx])for key, val in self.encodings.items()}
        item['go_hier_encodings'] = self.go_hier_encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ParsePassage:
    def __init__(self):
        try:
            self.pmid2content = pickle.load(open('./corpus/pmid2content.pickle', 'rb'))
        except FileNotFoundError:
            self.pmid2content = defaultdict(list)
            files = glob.glob(evidence_passages.format('*'))
            for file in tqdm(files):
                content = open(file, 'r').read().strip().split('\n\n')
                pmid = file[:-4].split('/')[-1]
                self.pmid2content[pmid] = content
            pickle.dump(self.pmid2content, open('./corpus/pmid2content.pickle', 'wb'))

    def get_title(self, pmid):
        return self.pmid2content[pmid][0]  # title

    def get_abstract(self, pmid):
        try:
            return self.pmid2content[pmid][1]  # abstract
        except IndexError:
            return ''  # case when text abstract do not contain abstract


class Encode:
    def __init__(self, x, max_length=350):
        print("encoding text")
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        try:
            self.encodings = self.tokenizers([Preprocessing(*i).forward() for i in x],
                                             max_length=max_length,
                                             padding=True,
                                             truncation=True)

        except TypeError:  # input is already paired, no preprocessing required
            self.encodings = self.tokenizers(x,
                                             max_length=max_length,
                                             padding=True,
                                             truncation=True)
        # self.labels = labels

    def forward(self):
        # dataset = DatasetBert(self.encodings, self.labels)
        return self.encodings

    def decode(self):
        for ids in self.encodings["input_ids"]:
            print(len(self.tokenizers.decode(ids).split(' ')))

    def mask_by_mention(self, evidence, go, alpha=0.5):
        """
        mask out GO term mentioned as keywords within evidence text
        :param evidence: str: evidence text
        :param go: str: go term text
        :param alpha: default 1, fraction of evidence text that need to be masked
        :return: go mention masked evidence and go
        """
        #tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        go_tokens = set(self.tokenizers.tokenize(go))
        evidence_tokens = self.tokenizers.tokenize(evidence)
        overlap_indice = []
        for idx in range(len(evidence_tokens)):
            if evidence_tokens[idx] in go_tokens:
                overlap_indice.append(idx)
                #evidence_tokens[idx] = ''

        # mask output overlaps
        for i in range(int(len(overlap_indice)*alpha//1)):
            evidence_tokens[overlap_indice[i]] = ''

        evidence = ' '.join(evidence_tokens).replace(' ##', '')
        return evidence, go


class FetchGOEmbedding:
    def __init__(self, go_hier_csv=go_hier):
        self.le = MakeGODAG(go_hier_csv).get_node_encoder()
        print(f"loading pre_trained GO KG embedding")
        self.kg_embeddings = GOKGModel().get_pre_trained()

    def get_kg_embedding(self, go_ids):
        indices = self.le.transform(go_ids)
        embeddings = []
        counter = 0  # counter no. of go terms out side of the pretrained go hier vocabulary
        for idx in indices:
            try:
                curr_embedding = self.kg_embeddings[idx]
            except IndexError:  # the hier info of current go term is not recorded in KG
                curr_embedding = torch.zeros(16)   # set OOV GO hier embedding as zero-tensor
                counter += 1
            embeddings.append(curr_embedding)
        print(f"GO KG shape is: {(len(embeddings), len(embeddings[0]))}")
        if counter:
            print(f'{counter} of go terms oov, plz update the pre_trained go embedding')
        return embeddings
