# PriCDSR

This is our implementation for our PriCDSR.
We use DASL as our base model.
We used the source code provided by the DASL authors at [https://github.com/lpworld/DASL](https://github.com/lpworld/DASL).

The scripts prefixed with *valid* are used for our hyperparameter search.

The scripts prefixed with *test* are used for our model comparison.

The scripts prefixed with *epsilon* are used for our parameter analysis.

The three files, **model.py**, **train.py**, and **utils.py**, come from the source code of DASL, and we have made some changes to them.

The file **add_noise_for_dasl.py** is used to add noise to the auxiliary dataset, which is the implementation of our proposed random mechanism.

The file **fix_length_for_raw_data.py** is used to fix the sequence length of the raw dataset.

## Requirements

We use the following software to run our codes.

- Python==3.6
- TensorFlow==1.14

## Quick Start

We use the same preprocessed dataset as that in MGCL, which is published at [https://csse.szu.edu.cn/staff/panwk/publications/MGCL/](https://csse.szu.edu.cn/staff/panwk/publications/MGCL/).

Please run **process.py** in its **cross_data** directory to obtain the processed data (which is stored in the **processed_data_all** directory), and then copy the **processed_data_all** to the current directory.

Then, please run **add_noise_for_dasl.py** and **fix_length_for_raw_data.py** to obtain data that can be used for our PriCDSR with and without adding different levels of noise, respectively.

After that, you can run any script prefixed with *valid* for hyperparameter search, any script prefixed with *test* for model comparison, and any script prefixed with *epsilon* for parameter analysis.

Taking the dataset Movie $\leftarrow$ Book as an example, you can run the scripts **valid-mb.sh**, **test-mb.sh**, and **epsilon-mb-10.sh**.

## Partial results

The following table shows the recommendation performance of the single-domain sequential recommendation (SR) method DASL-single, cross-domain SR (CDSR) method DASL, and our privacy-preserving CDSR method PriCDSR-DASL on dataset Movie $\leftarrow$ Book.

|Algorithms|HR@5|HR@10|NDCG@5|NDCG@10|
|:-|:-:|:-:|:-:|:-:|
|DASL-single|0.1754 $\pm$ 0.0013|0.2701 $\pm$ 0.0040|0.1164 $\pm$ 0.0016|0.1468 $\pm$ 0.0024|
|DASL|0.1903 $\pm$ 0.0029|0.2860 $\pm$ 0.0042 | 0.1269 $\pm$ 0.0025 | 0.1578 $\pm$ 0.0029 |
|PriCDSR-DASL|0.1791 $\pm$ 0.0023 | 0.2755 $\pm$ 0.0030 | 0.1174 $\pm$ 0.0013 | 0.1484 $\pm$ 0.0016 |

The following table shows the recommendation performance of our PriCDSR-DASL with different privacy budget $\epsilon$.

|Algorithms|HR@5|HR@10|NDCG@5|NDCG@10|
|:-|:-:|:-:|:-:|:-:|
| PriCDSR-DASL($\epsilon=1$)  | 0.1713 $\pm$ 0.0014 | 0.2659 $\pm$ 0.0055 | 0.1123 $\pm$ 0.0007 | 0.1426 $\pm$ 0.0009 |
| PriCDSR-DASL($\epsilon=2$)  | 0.1761 $\pm$ 0.0023 | 0.2751 $\pm$ 0.0054 | 0.1160 $\pm$ 0.0028 | 0.1478 $\pm$ 0.0034 |
| PriCDSR-DASL($\epsilon=5$)  | 0.1750 $\pm$ 0.0020 | 0.2680 $\pm$ 0.0016 | 0.1155 $\pm$ 0.0027 | 0.1453 $\pm$ 0.0025 |
| PriCDSR-DASL($\epsilon=10$) | 0.1791 $\pm$ 0.0023 | 0.2755 $\pm$ 0.0030 | 0.1174 $\pm$ 0.0013 | 0.1484 $\pm$ 0.0016 |
| PriCDSR-DASL($\epsilon=20$) | 0.1920 $\pm$ 0.0023 | 0.2886 $\pm$ 0.0031 | 0.1277 $\pm$ 0.0018 | 0.1587 $\pm$ 0.0022 |
| PriCDSR-DASL($\epsilon=50$) | 0.1856 $\pm$ 0.0086 | 0.2815 $\pm$ 0.0107 | 0.1239 $\pm$ 0.0048 | 0.1548 $\pm$ 0.0055 |
| DASL                        | 0.1903 $\pm$ 0.0029 | 0.2860 $\pm$ 0.0042 | 0.1269 $\pm$ 0.0025 | 0.1578 $\pm$ 0.0029 |
