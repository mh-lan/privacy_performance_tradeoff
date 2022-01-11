# Decentralized learning with data exchanging[![DOI](https://zenodo.org/badge/446033567.svg)](https://zenodo.org/badge/latestdoi/446033567)

This is the project code for course COMP7203P.01.2021FA in USTC.

We are interested in a question: Will privacy leakage improve performance in decentralized learning?

In our setting, each client owns non-i.i.d. training data (e.g., each client only has images of 2 digits). Before decentralized learning, each client sends its own data to the neighbors. This action can decrease non-i.i.d. level of training data, and thus improve the overall performance. However, their data privacy will also expose to the neighbors via this action. Find a trade-off between privacy leakage and learning performance is what we want.

We examine the privacy leakage as KL divergence between the distribution of transmitted images and neighbor's images. Through the scripts in this reposiry, experiments with different client topology, initial number of training digits, ratio of transmitted data, neral network model can be down. 

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.8，

pytorch>=1.10

## Run

Federated learning with MLP and CNN is produced by:
> python [main.py](main.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main.py --digit_num 3 --share_ratio 0.1 --aggr_model de --model cnn --epochs 50 --gpu 0  

The result are recorded in result.xlsx.

## Result

See it in [report.pdf](report.pdf)


## Acknowledgements
Acknowledgements give to Min Guo and Zhiwei Yao.

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

## Cite As

Muhang Lan. (2022). Decentralized learning with data exchanging (v1.0). Zenodo. https://doi.org/10.5281/zenodo.5835914
