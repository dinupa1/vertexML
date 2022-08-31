# VertexML

A machine learning approach to find the vertex information of the reconstructed tracks.

This module is based on the [`particleflow`](https://github.com/jpata/particleflow) framework.


## Data Creation
Raw data was created using the [`SimChainDev`](https://github.com/E1039-Collaboration/e1039-analysis/tree/master/SimChainDev) of the `e1030-analysis` framework. Raw data contains `~20k` Drell-Yan events with $\mu^{+}$ particles and you can find it in the `data/raw` directory. The structure of the input data is;
```
X = [charge, x, y, z, px, py, pz]
Y = [x, y, z]
```

If you want to create a custom data the change the `ProcessData.py` module in the `source` directory. Then run the `process_data.sh` file to save the raw data to `.pt` format in the `data/processed` directory.

## Testing the Module

It is highly recommended to use the [`conda`](https://github.com/conda-forge/miniforge) environment to test this module.
```bash
git clone https://github.com/dinupa1/vertexML.git
cd vertexML
conda env create -f environment.yml
conda activate vertexML
python3 setup.py install
```

Alternatively you can use the `pip` to install the module.

You can run the `analysis.ipynb` as an example.

