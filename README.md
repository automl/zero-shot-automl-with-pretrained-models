# Zero-Shot AutoML with Pretrained Models
![Overview](https://raw.githubusercontent.com/automl/zero-shot-automl-with-pretrained-models/master/overview.png)
Given a new dataset D and a low compute budget, how should we choose a pre-trained model to fine-tune to D, and set the fine-tuning hyperparameters without risking overfitting, particularly if D is small? Here, we extend automated machine learning (AutoML) to best make these choices. Our domain-independent meta-learning approach learns a zero-shot surrogate model which, at test time, allows to select the right deep learning (DL) pipeline (including the pre-trained model and fine-tuning hyperparameters) for a new dataset D given only trivial meta-features describing D such as image resolution or the number of classes. To train this zero-shot model, we collect performance data for many DL pipelines on a large collection of datasets and meta-train on this data to minimize a pairwise ranking objective. We evaluate our approach under the strict time limit on the vision track of the ChaLearn AutoDL challenge benchmark, clearly outperforming all challenge contenders.

Paper link: tba

# Download Models and Meta-Dataset
All model files and the meta-dataset can be downlaoded here: https://www.dropbox.com/sh/1yvk1034zkw4o0d/AAA8Kq5tTgrjK-ro4wCTWVNga?dl=0

# Installation


# Documentation
