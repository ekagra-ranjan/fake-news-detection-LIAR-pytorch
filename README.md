# Fake News Detection by Learning Convolution Filters through Contextualized Attention

## Dataset
The [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset) consists of 12,836 short statements taken from POLITIFACT and labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false, half-true, mostly-true, and true. These six label sets are relatively balanced in size. The statements were collected from a variety of broadcasting mediums, like TV interviews, speeches, tweets, debates, and they cover a broad range of topics such as the economy, health care, taxes and election. The [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset is an extension to the LIAR dataset by automatically extracting for each claim the justification that humans have provided in the fact-checking article associated with the claim.

## Network Architecture
![Screenshot 1](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/fake-net.png "Net")


## Methodology
Instead of directly extracting features from Statement, we employ an attention mechanism to use the given side information (subject, speaker, job, state, party, context and justification) to attend over the given statement to check its truthfulness. The attention mechanism makes the process of feature extraction from statement contextualized based on side information. See Fig. 1 for the graphical representation of
the architecture. For more detailed explanation of the approach read the [paper](https://www.researchgate.net/publication/341378920_Fake_News_Detection_by_Learning_Convolution_Filters_through_Contextualized_Attention).

## How to Use

Run `main.py` which is the driver of the experiments. To train a model change the variable `mode` in `main.py` to `train`. For evaluating a saved model, change `mode` to `test` and put the name of the saved model in the variable `pathModel`. To run LIAR dataset, change the variable `dataset_name` to `LIAR` and if you want to run LIAR-PLUS dataset then change `dataset_name` to `LIAR-PLUS`.


# Citation:
Please cite the paper if you found it useful in your work.
```bibtex
@unknown{unknown,
author = {Ranjan, Ekagra},
year = {2019},
month = {08},
pages = {},
title = {Fake News Detection by Learning Convolution Filters through Contextualized Attention},
doi = {10.13140/RG.2.2.20829.84968}
}
```


## Acknowledgement
I would like to thank [FangJun Zhang](https://github.com/zfjmike) for open-sourcing the code for LIAR dataset in his [repo](https://github.com/zfjmike/fake-news-detection) which served as the starting point for my work.
