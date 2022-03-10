# AutoGOAConsistency
 Source code of PubMedBERT and GNN-BERT for Paper `Literature-based Gene Ontology Annotation`


## Installation
1) Download zip package of this project.

2) Install required dependencies shown in `requirements.txt`

3) GO to one of directories `/bioconsis-baseline` for utilizing PubMedBERT or `/bioconsis-gnn` for utilizing GNN-BERT

___
## Evaluation of PubMedBERT and GNN-BERT models
### To evaluate PubMedBERT model (pre-trained with in-distribution samples)
1) Go to `/bioconsis-baseline` directory
2) Execute `python3 evaluation -b 0`
3) A list of predicted GOA inconsistency class will be generated in `/output/pubmedbert.txt`
4) A confusion matrix, $Precision$, $Recall$, $F_1$ will be displayed on the screen



### To evaluate GNN-BERT model (pre-trained with in-distribution samples)
1. Go to `/bioconsis-gnn directory`
2. Execute `python3 evaluation -b 0`
3. A list of predicted GOA inconsistency class will be generated in `/output/gnnbert.txt`
4. A confusion matrix, $Precision$, $Recall$, $F_1$ will be displayed on the screen

___

## Interpretation of models's output
The GOA self-consistency and four types of inconsistency is encoded in numeric values ranging from $0-5$

<details><summary>Details</summary>
<p>

         CO (self-consistency) -> 0 
         OS (over-specific) -> 1
         OB (over-broad) -> 2
         IM (irrelevant GO mention) -> 3
         IG (incorrect gene product) -> 4

</p>
</details>

___
## Training of new PubMedBERT and GNN-BERT models
1) GO inside of `/corpus` directory, construct your own training set and development set in `*.csv` as demonstration of `imbal_bc4go_calibrated_score.csv`.
2) GO to home directory, edit `corpus_path.py' and replace the following two parameters with your newly created dataset path

`train_csv=[*csv path to your training set]`

`dev_csv=[*csv path to your dev set]`

3) Add new go info in `/corpus/go_info.csv` if any GOId in your dataset is missing within existing resources; add new gene info in `/corpus/gene_info.csv` if any GeneId in your dataset is missing within existing resources; load `pmid2content.pickle` as dictionary and add new PMID->TitleAbstract map for update if any PMID in your dataset is missing within exisiting resources.


4) Run python3 modelling.py -b 0 to start the training



## Interpretation of GNN-BERT
Go to `/bioconsis-gnn` where you will find `knowledge-graph.py`, demonstrating the implementation of GNN for encoding GO hierarchical knowledge.

`joint.py`, demonstrating the concatenation of pre-trained GNN vertex encoding and PubMedBERT embeddings.

`sage_go_kg_64dim.pt` is the pre-trained GO hierarchical vertex encoding. 

