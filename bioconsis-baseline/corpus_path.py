# train_csv = './corpus/*.csv'  # create your own training set
dev_csv = './corpus/balanced_dev.csv'  # development set
test_csv = './corpus/imbal_bc4go_calibrated_score.csv'  # test set


# text attribute lookup
go_info = './corpus/go_info_large.csv'  # store GO related info
gene_info = './corpus/gene_info.csv'  # store gene related info


# text passages with title and abstract, indexed by PMID
evidence_passages = './corpus/pmid2content.pickle'
