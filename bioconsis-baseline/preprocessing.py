import torch
import glob
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd
from corpus_path import evidence_passages
from corpus_path import go_info, gene_info
from tqdm import tqdm
from textblob import TextBlob

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
        try:
            self.abstract = str(TextBlob(str(abstract)).sentences[-1])
        except:
            self.abstract = ''
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
        #return mask_by_mention(gene_conditioned_evidence, go_info)  # mask go concept
        return gene_conditioned_evidence, go_info


class DatasetBert(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, binary=False):
        print("converting to BERT dataset format")
        self.encodings = encodings
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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
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
    def __init__(self, x):
        print("encoding text")
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.encodings = self.tokenizers([Preprocessing(*i).forward() for i in x],
                                         max_length=350,
                                         padding=True,
                                         truncation=True)
        # self.labels = labels

    def forward(self):
        # dataset = DatasetBert(self.encodings, self.labels)
        return self.encodings

    def decode(self):
        for ids in self.encodings["input_ids"]:
            print(len(self.tokenizers.decode(ids).split(' ')))

    def mask_by_mention(self, evidence, go, alpha=0.7):
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
