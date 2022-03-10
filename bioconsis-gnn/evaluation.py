import getopt
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from joint import JointModelForSequenceClassification
from knowledge_graph import GOKGModel, SAGE, MLPPredictor
import time
from corpus_path import test_csv
from preprocessing import Integration, Encode, DatasetBert, FetchGOEmbedding
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy

def compute_auc(labels, predictions_logits):
    label_name = ['Consistent', 'Over_specific', 'Over_broad', 'Irr-go mention', 'Err-gene']
    labels = label_binarize(labels, classes=[0, 1, 2, 3, 4])
    n_classes = 5
    m = torch.nn.Softmax()
    y_score = m(torch.from_numpy(predictions_logits))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: {label_name[i]} vs Rests')
        plt.legend(loc="lower right")
        #plt.savefig(f'./output/roc_gnn_calibrated_{i}.pdf')
        #plt.show()

    # precision recall curve
    precision = dict()
    recall = dict()
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='{}'.format(label_name[i]))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    #plt.savefig('./output/prc_gnn_calibrated.pdf')


def evaluation_summary(y_pred, y_test):
    """
    summary of accuracy, macro presicion,
    recall, f1 score
    """
    print("Accuracy:")
    print(metrics.accuracy_score(y_test, y_pred))

    print("\n Micro Average precision:")
    print(metrics.precision_score(y_test, y_pred, average='micro'))

    print("\n Micro Average recall:")
    print(metrics.recall_score(y_test, y_pred, average='micro'))

    print("\n Micro Average f1:")
    print(metrics.f1_score(y_test, y_pred, average='micro'))

    print("\n Classification report:")
    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':
    BINARY = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:", ["binary="])
    except getopt.GetoptError:
        print('evaluation.py -b <True:False>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluation.py -b <True:False>')
            sys.exit(2)
        elif opt in ("-b", "--binary"):
            BINARY = True if arg=='True' else False

    if BINARY:
        NUM_OF_CLASS = 2
    else:
        NUM_OF_CLASS = 5

    test_integration = Integration(test_csv)
    test_labels = [l - 1 for l in pd.read_csv(test_csv)['edge_type'].tolist()]
    test_x = test_integration.forward()  # integrate text attributes
    test_encodings = Encode(test_x).forward()  # encode text
    test_go_embeddings = FetchGOEmbedding().get_kg_embedding(pd.read_csv(test_csv)['des'].tolist())
    test_dataset = DatasetBert(test_encodings, test_labels, test_go_embeddings, binary=BINARY)  # formatting dataset
    #test_dataset = DatasetBert(test_encodings, test_labels, binary=BINARY)  # format dataset
    # dev_dataset = train_dataset
    print(f'{test_dataset.__len__()} instances in test set')

    # load self-defined pre_trained BERT-KG model
    model = JointModelForSequenceClassification.from_pretrained("./injections/output/10p_inject_gnn_train",
                                                               num_labels=NUM_OF_CLASS)
    print(f'the number of classes is {NUM_OF_CLASS}')

    # training args are not required in testing stage
    training_args = TrainingArguments(
        output_dir='./output/results',  # output directory
        #num_train_epochs=3,  # total number of training epochs
        #per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        #warmup_steps=300,  # number of warmup steps for learning rate scheduler
        #weight_decay=0.01,  # strength of weight decay
        #logging_dir='./output/logs',  # directory for storing logs
        #logging_steps=10,
    )

    # create BERT fine-tuner
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
    )

    # begin evaluation
    y_pred = trainer.predict(test_dataset)
    preds = np.argmax(y_pred.predictions, axis=1)
    evaluation_summary(preds, test_labels)
    compute_auc(test_labels, y_pred.predictions)
    
    # write predicted class to txt file
    fname = '10_results'#time.time()
    with open(f'./injections/output/{fname}.txt','w') as f:
        f.write('\n'.join(str(x) for x in preds))
        print(f'predictions were successfully saved to ./output/{fname}.txt')
    f.close()
    
    with open(f'./injections/output/entropy_{fname}.txt','w') as f:
        f.write('\n'.join(str(x) for x in [entropy(softmax(logits),base=2) for logits in y_pred.predictions]))
        print(f'entropy was saved to ./output/entropy_{fname}.txt')
    f.close()
    
