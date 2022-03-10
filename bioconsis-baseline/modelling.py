from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from corpus_path import train_csv, dev_csv
import pandas as pd
from preprocessing import Integration, Encode, DatasetBert
import time
import sys, getopt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if __name__ == '__main__':
    BINARY = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:", ["binary="])
    except getopt.GetoptError:
        print('modelling.py -b <True:False>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('modelling.py -b <True:False>')
            sys.exit(2)
        elif opt in ("-b", "--binary"):
            BINARY = True if arg=='1' else False

    if not BINARY:
        NUM_OF_CLASS = 5
    else:
        NUM_OF_CLASS = 2
    # train dataset
    train_integration = Integration(train_csv)
    train_labels = [l - 1 for l in pd.read_csv(train_csv)['edge_type'].tolist()]  # formatting label encoding
    train_x = train_integration.forward()  # integrate text attributes
    train_encodings = Encode(train_x).forward()  # encode text
    train_dataset = DatasetBert(train_encodings, train_labels, binary=BINARY)  # formatting dataset
    print(f'{train_dataset.__len__()} instances in training set')

    # eval dataset
    dev_integration = Integration(dev_csv)
    dev_labels = [l - 1 for l in pd.read_csv(dev_csv)['edge_type'].tolist()]
    dev_x = dev_integration.forward()  # integrate text attributes
    dev_encodings = Encode(dev_x).forward()  # encode text
    dev_dataset = DatasetBert(dev_encodings, dev_labels, binary=BINARY)  # format dataset
    print(f'{dev_dataset.__len__()} instances in dev set')

    # load pre_trained model
    if not BINARY:
        model = AutoModelForSequenceClassification.\
            from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels=NUM_OF_CLASS)
    else:#microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        model = AutoModelForSequenceClassification. \
            from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels=NUM_OF_CLASS)
    print(f'the number of classes is {NUM_OF_CLASS}')


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        #with open('./injections/output/metrics_30p.txt', 'a') as f:
        #    f.write('\t'.join((str(precision), str(recall), str(f1), str(acc))) + '\n')
        #f.close()
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # set training args
    training_args = TrainingArguments(
        output_dir='./output/results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        evaluation_strategy="steps",  # evaluate at every step
        logging_dir='./output/logs',  # directory for storing logs
        logging_steps=25,
    )

    # create BERT fine-tuner
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics = compute_metrics  # evaluation score
    )

    # begin fine-tuning
    trainer.train()
    trainer.save_model("./output/evidence_last_sent")
