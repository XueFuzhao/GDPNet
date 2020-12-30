# GDPNet

GDPNet: Refining Refining Latent Multi-View Graph for Relation Extraction

The code is divided into two parts, DialogRE and TACRED, by datasets.

# Requirements

PyTorch == 1.4

CUDA == 10.1

Apex

We perform our experiments on GTX 1070 GPU. Please use the same hardware and software environment if possible to ensure the same results.

The model on GTX 1070 GPU has been released: https://drive.google.com/drive/folders/1eod1YmMP6pcVU4-7vTIIqE0oVOUrBERG?usp=sharing

We also reproduced our results on Q8000 GPU, and V100 GPU. 

The model on Q8000 GPU has been released: https://drive.google.com/drive/folders/1CMXmO8_hqB1L_Z6zfJuQ2g56tKqTq_y7?usp=sharing

The model on V100 GPU has been released: https://drive.google.com/drive/folders/1lU5SAmclacFtgqX5I3qmhGXR1RnZoAZg?usp=sharing


# DialogRE

This dataset can be downloaded at: https://github.com/nlpdata/dialogre

Download and unzip BERT from https://github.com/google-research/bert, and set up the environment variable for BERT by export BERT_BASE_DIR=/PATH/TO/BERT/DIR in every run_GDPNet.sh.

We also provide the BERT-base-uncased (PyTorch Version): https://drive.google.com/drive/folders/1qBzjWDVpXSBXfmxO6yW6ATBX3D7LW5YZ?usp=sharing

(1) Please copy the *.json files into DialogRE/data

(2) Train the GDPNet model
```sh
$ cd GDPNet
$ bash run_GDPNet.sh
```

Note: we also provided the logits_dev.txt and logits_test.txt, so we can run the last line of run_GDPNet.sh to see the results directly. Please copy the files in https://drive.google.com/drive/folders/1CMXmO8_hqB1L_Z6zfJuQ2g56tKqTq_y7?usp=sharing into GDPNet/DialogRE/GDPNet



# TACRED

TACRED URL: https://nlp.stanford.edu/projects/tacred/

TACRED-Revisit URL: https://github.com/DFKI-NLP/tacrev/

(1) Please download the TACRED and TACRED-Revisit and copy them into GDPNet/tacred and GDPNet/tacred_revisit respectively.

    TACRED URL: https://nlp.stanford.edu/projects/tacred/
    
    TACRED-Revisit URL: https://github.com/DFKI-NLP/tacrev/

(2) Train the GDPNet model

```sh
$ cd GDPNet
$ bash run.sh
```

Note: The default dataset is TACRED, pls change the --data_dir in run.sh to try TACRED-revisit


