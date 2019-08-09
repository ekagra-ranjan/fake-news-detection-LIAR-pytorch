# fake-news-detection-LIAR-pytorch

## Dataset
[LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) is a benchmark dataset for fake news detection, released recently. This dataset has evidence sentences extracted automatically from the full-text verdict report written by
journalists in Politifact. It consists of 12,836 short statements taken from POLITIFACT and
labeled by humans for truthfulness, subject, context/venue, speaker, state, party, and prior
history. For truthfulness, the LIAR dataset has six labels: pants-fire, false, mostly-false,
half-true, mostly-true, and true. These six label sets are relatively balanced in size.

## Network Architecture
![Screenshot 1](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/fake-net.png "Net")


## Methodology
Instead of directly extracting features from Statement, we employ an attention mechanism to use the given side information (subject, speaker, job, state, party, context and justification) to attend over the given statement to check its truthfulness. The attention mechanism makes the process of feature extraction from statement contextualized based on side information. See Fig. 1 for the graphical representation of
the architecture. For more detailed explanation of the approach read the [report](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch/blob/master/report.pdf)

## Acknowledgement
I would like to thank [FangJun Zhang](https://github.com/zfjmike) for open-sourcing the code for LIAR dataset in his [repo](https://github.com/zfjmike/fake-news-detection) which served as the starting point for my work.
