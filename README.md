## Quora Question Pairs — Graph-Enhanced NLP Ensemble (0.1268 LogLoss)

Problem Overview:
  The Quora Question Pairs (QQP) task aims to determine whether two natural-language
questions are semantically equivalent. The dataset contains noisy labels,
paraphrase variation, and strong lexical overlap biases, making calibration and
robust feature engineering critical.

Motivation:
  This is an old Kaggle competetion, but still very classical till today. Just wnated to try
  this it with some modern transformer model and see where I can reach on the LB

## Key ideas:
  
    - lexical overlap
    - token to token alignment
    - local pattern exploration over token level alignment map
    - bi-encoder embedding comparison
    - cross encoder attention
    - topic distribtuion difference
    - graph density group

## Features:

    - **Neural Models**
        - SBERT
        - ESIM / DIIN / BiMPM for fine-grained token interactions
        - DeBERTa-v3 cross-encoder
    - **Graph Features**
        - Question graph built from train + test (unlabeled) edges
        - Degree, Adamic-Adar, Katz, PageRank, multi-hop neighbors, components, triangle
        - Node2Vec embeddings
    - **Topic Models**
        - NMF
        - LDA/LSA
        - tfidf
        - BTM
    - **Feature-Level Ensembling**
        - Embedding interaction statistics (cos / L1 / L2 / prod)
        - Graph + neural + lexical features stacked via LightGBM

## Hardware:

  I used single NVIDA GeForce RTX 4070 SUPER 12GB VRAM and 64GB RAM which is a very moderate hardware setting
  that most of you should have

## How reproduce the solution:
  1. Download the datasets from Kaggle to `data/`
  2. Install the requirements.txt
  3. Run `download_fasttext.sh` and `download_glove.sh` to download the glove and fasttext embedding if you don't have them
  4. Create a folder named `artifacts/`, `artifacts/training`, `artifacts/prediction` first
  5. Run each script inside `handcraft/`
  6. Run each script inside `graph/`
  7. Run each script inside `models/`
  8. run `stacking.py` at the root directory which will produce a submission.csv inside `artifacts/`

  All scripts are independent of each other and the features are stored inside artifacts and its respective subfolders as
  features for stacking model


