# fake-news-detection-LIAR-pytorch

## Dataset
The [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset) consists of 12,836 short statements taken from POLITIFACT and labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false, half-true, mostly-true, and true. These six label sets are relatively balanced in size. The statements were collected from a variety of broadcasting mediums, like TV interviews, speeches, tweets, debates, and they cover a broad range of topics such as the economy, health care, taxes and election. The [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) dataset is an extension to the LIAR dataset by automatically extracting for each claim the justification that humans have provided in the fact-checking article associated with the claim.

## Network Architecture
![Screenshot 1](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/fake-net.png "Net")


## Methodology
Instead of directly extracting features from Statement, we employ an attention mechanism to use the given side information (subject, speaker, job, state, party, context and justification) to attend over the given statement to check its truthfulness. The attention mechanism makes the process of feature extraction from statement contextualized based on side information. See Fig. 1 for the graphical representation of
the architecture. For more detailed explanation of the approach read the [report](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/report.pdf).

## Acknowledgement
I would like to thank [FangJun Zhang](https://github.com/zfjmike) for open-sourcing the code for LIAR dataset in his [repo](https://github.com/zfjmike/fake-news-detection) which served as the starting point for my work.
