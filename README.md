# AutoGOAConsistency
 Source code of PubMedBERT and GNN-BERT for Paper `Exploring Automatic Inconsistency Detection for
Literature-based Gene Ontology Annotation`


## Installation
1) Clone this project into local file as a directory

2) Download pre-trained PubMedBERT, GNN-BERT, and literature title/abstract content [here](https://www.dropbox.com/sh/jd3403hhvt1uu3q/AADOB38VWAWyz4RkhsthNa_Ca?dl=0)
<details><summary>Details</summary>
<p>

         unzip `baseline_pre_fine_tuned.zip` and place it in `/bioconsis-baseline/output`

         unzip `universal_fine_tuned.zip` and place it in `/bioconsis-gnn/output`

         place `pmid2content.pickle` in both `/bioconsis-baseline/corpus` and `/bioconsis-gnn/corpus`

</p>
</details>

3) Install any missing dependency as listed in `requirements.yml`

4) GO to one of directories `/bioconsis-baseline` for utilizing PubMedBERT or `/bioconsis-gnn` for utilizing GNN-BERT

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
2) GO back to `/bioconsis-baseline/` or `/bioconsis-gnn`; Edit `corpus_path.py' by replacing the following two parameters with your newly created dataset path

`train_csv=[path to your training set *csv]`

`dev_csv=[path to your dev set *csv]`

3) Add new go info in `/corpus/go_info.csv` if any GOId in your dataset is missing within `go_info.csv`; Add new gene info in `/corpus/gene_info_large.csv` if any GeneId in your dataset is missing within `gene_info_large.csv`; Load `pmid2content.pickle` as dictionary and add new PMID->TitleAbstract map for update if any PMID in your dataset is missing.


4) Run python3 modelling.py -b 0 to start the training of PubMedBERT in `/bioconsis-baseline/` or the training of GNN-BERT in `/bioconsis-gnn/`



## Interpretation of GNN-BERT
Go to `/bioconsis-gnn` where `knowledge-graph.py` locates, demonstrating the implementation of GNN for encoding GO specificity knowledge.

`joint.py`, demonstrating the concatenation of pre-trained GNN vertex encoding and PubMedBERT embeddings.

`sage_go_kg_64dim.pt`, the pre-trained GO specificty vertex encoding.