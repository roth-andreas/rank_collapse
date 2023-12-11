# Repository

This repository contains the implementation for "Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks" that was published at the Learning on Graphs (LoG) Conference 2023.

## Dependencies

This code requires the following dependencies to be installed:

* Python > 3
* PyTorch > 2.0
* PyTorch Geometric >= 2.3
* gdown
* matplotlib


## Usage

### Node Classification Datasets

To run the experiments, execute "main.py" inside the node classification folder with the following arguments:

* --convs={KP,softmax_SKP,SKP} Choose the desired models.
* --datasets={Cora, Citeseer, Pubmed,texas,cornell,wisconsin,film,chameleon,squirrel} Choose the desired datasets.
* --h_dim Hidden dimension of the model
* --layers Number of layers

### Synthetic Dataset

To run the experiment that compares the Dirichlet energy and the norm, simply execute "norm_vs_dirichlet_energy.py". Figures will be directly generated.

To run the experiment comparing the performance of KP, softmax-SKP, and SKP on random graphs, run "main.py" inside the synthetic folder with the following arguments:

* --conv={gat,fagcn,skp} Choose the desired model.
* -ho Set this when the aggregation matrices for all layers should be homogeneous.
* -li Set this when no activation function should be applied
* --num_graphs Number of graphs to repeat the experiment for
* --start_graph Starting seed of the initial graph

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{roth2023rank,
  title={Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks},
  author={Roth, Andreas and Liebig, Thomas},
  booktitle={Learning on Graphs Conference},
  year={2023},
  organization={PMLR}
}

