# GRAPHDeep

A comprehensive study of graph deep learning enabled spatial domains discrimination technologies for spatial transcriptomics
------
Graph deep learning has been regarded as a promising methodology to address genetic transcriptomes and spatial locations in spatial omics data. To this end, a comprehensive analytical toolbox, GRAPHDeep, is presented to aggregate two graph deep learning modules (i.e., Variational Graph Auto-Encoder and Deep Graph Infomax) and twenty graph neural networks for spatial domains discrimination. Towards spatial omics data with various modalities and scales, the best integration of graph deep learning module and graph neural network is determined. Consequently, this built framework can be regarded as desirable guidance for choosing an appropriate graph neural network for heterogeneous spatial data.
## Contents
* [Prerequisites](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#prerequisites)
* [Example usage](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#example-usage)
* [Trouble shooting](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#trouble-shooting)

### Prerequisites

1. Python (>=3.8)
2. Scanpy
3. Squidpy
4. Pytorch_pyG
5. pandas
6. numpy
7. sklearn
8. seaborn
9. matplotlib
10. torch_geometric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example usage
* Selecting GNNs for spatial clustering task in DGI module
  ```sh
    running DGI Example_DLPFC.ipynb to see the simulation results step by step
  ```
* Selecting GNNs for spatial clustering task in VGAE module
  ```sh
    running VGAE Example_DLPFC.ipynb to see the simulation results step by step
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Trouble shooting

* data files<br>
Please down load the spatial transcriptomics data from the provided links.

* Porch_pyg<br>
Please follow the instruction to install pyG and geometric packages.
