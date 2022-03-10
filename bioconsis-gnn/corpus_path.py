# GOA link csv
"""
train_csv = './corpus/large_balanced_train.csv'
dev_csv = './corpus/balanced_dev.csv'
test_csv = './corpus/imbal_bc4go_calibrated_score.csv'
"""
train_csv = './injections/5p_inject_base_train.csv'
dev_csv = './corpus/balanced_dev.csv'
test_csv = './corpus/imbal_bc4go_calibrated_score.csv'  # './corpus/bc4go_calibrated_score.csv'

test_purpose = './corpus/test_model.csv'

# text attribute lookup
go_info = './corpus/go_info_large.csv'
gene_info = './corpus/gene_info.csv'

go_hier = './corpus/go_network.csv'

# text passages with title and abstract
evidence_passages = './corpus/articles/{}.txt'

# knowledge graph node embeddings
#go_kg_embeddings = './sage_go_kg.pkl'
