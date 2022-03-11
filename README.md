# AutoGOAConsistency
Implementation of PubMedBERT and GNN-BERT in paper (accessible soon):
> Exploring Automatic Inconsistency Detection for Literature-based Gene Ontology Annotation, Jiyu Chen, Benjamin Goudey, Justin Zobel, Nicholas Geard, Karin Verspoor (Submitted)

## Overview

**Motivation:** Literature-based Gene Ontology Annotations (GOA) are biological database records that use controlled vocabulary to uniformly represent gene function information that is described in primary literature. Assurance of the quality of GOA is crucial for supporting biological research. However, a range of different kinds of inconsistencies in between literature as evidence and annotated GO terms can be identified; these have not been systematically studied at record level. The existing manual-curation approach to GOA consistency assurance is inefficient and is unable to keep pace with the rate of updates to gene function knowledge. Automatic tools are therefore needed to assist with GOA consistency assurance. This paper presents an exploration of different GOA inconsistencies and an early feasibility study of automatic inconsistency detection.

**Results:** We have created a reliable synthetic dataset to simulate four realistic types of GOA inconsistency in biological databases. Three automatic approaches are proposed. They provide reasonable performance on the task of distinguishing the four types of inconsistency and are directly applicable to detect inconsistencies in real-world GOA database records. Major challenges resulting from such inconsistencies in the context of several specific application settings are reported.

**Conclusion:** This is the first study to introduce automatic approaches that are designed to address the challenges in current GOA quality assurance workflows.


## Installation
1) Clone this project into your local file system and unzip as a directory

2) Download three dependent resources from Dropbox, including a pre-trained PubMedBERT, GNN-BERT, and literature title/abstract content [here](https://www.dropbox.com/sh/jd3403hhvt1uu3q/AADOB38VWAWyz4RkhsthNa_Ca?dl=0). Place them in corresponding directories as follows:
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
### To evaluate PubMedBERT model
1) Go to `/bioconsis-baseline` directory
2) Execute `python3 evaluation -b 0`
3) A list of predicted GOA inconsistency class will be generated in `/output/pubmedbert.txt`
4) A confusion matrix, $Precision$, $Recall$, $F_1$ will be displayed on the screen



### To evaluate GNN-BERT model
1. Go to `/bioconsis-gnn directory`
2. Execute `python3 evaluation -b 0`
3. A list of predicted GOA inconsistency class will be generated in `/output/gnnbert.txt`
4. A confusion matrix, $Precision$, $Recall$, $F_1$ will be displayed on the screen

___

## Interpretation of models's output
Predictions are stored in `output/pubmedbert.txt` for PubMedBERT and `output/gnnbert.txt` for GNN-BERT. Predictions include GOA self-consistency and four types of inconsistency, encoded in numerical values ranging from $0-4$

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
1) GO to `/corpus` directory, construct your own training set and development set in `*.csv` as demonstration of `imbal_bc4go_calibrated_score.csv`.
2) GO back to `/bioconsis-baseline/` or `/bioconsis-gnn`; Edit `corpus_path.py' by replacing the following two parameters with your newly created dataset path

`train_csv=[path to your training set *csv]`

`dev_csv=[path to your dev set *csv]`

3) Add new go info in `/corpus/go_info.csv` if any GOId in your dataset is missing within `go_info.csv`; Add new gene info in `/corpus/gene_info_large.csv` if any GeneId in your dataset is missing within `gene_info_large.csv`; Load `pmid2content.pickle` as dictionary and add new PMID->TitleAbstract map for update if any PMID in your dataset is missing.


4) Run python3 modelling.py -b 0 to start the training of PubMedBERT in `/bioconsis-baseline/` or the training of GNN-BERT in `/bioconsis-gnn/`

___
## Application of PubMedBERT and GNN-BERT models
1) Place the GOA records in `/corpus/` directory in `*.csv` format as demonstration of `imbal_bc4go_calibrated_score.csv`.

2) Randomly assign `edge_type` column as numerical number ranging from $0-4$. These are placeholders of labels, which will not influence the application of the models.

3) Edit `corpus_path.py' by replacing the `test_csv=` with your newly created dataset path

4) Add new go info in `/corpus/go_info.csv` if any GOId in your dataset is missing within `go_info.csv`; Add new gene info in `/corpus/gene_info_large.csv` if any GeneId in your dataset is missing within `gene_info_large.csv`; Load `pmid2content.pickle` as dictionary and add new PMID->TitleAbstract map for update if any PMID in your dataset is missing.

5) Run python3 modelling.py -b 0 to start the training of PubMedBERT in `/bioconsis-baseline/` or the training of GNN-BERT in `/bioconsis-gnn/`

___
## Interpretation of GNN-BERT source code
1) Go to `/bioconsis-gnn` where `knowledge-graph.py` locates, demonstrating the implementation of GNN with two GraphSAGE layers for encoding GO specificity knowledge.

2) `joint.py`, demonstrating the concatenation of pre-trained GNN vertex encoding and PubMedBERT embeddings.

3) `sage_go_kg_64dim.pt`, the pre-trained GO specificty vertex encoding.