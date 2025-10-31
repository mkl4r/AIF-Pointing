# AIF-Pointing

Repository accompanying the paper ["An Active Inference Model of Mouse Point-and-Click Behaviour"](https://arxiv.org/abs/2510.14611) presented at the 6th International Workshop on Active Inference.

## Installation

### Prerequisites
- Python ≥ 3.11
- JAX (CPU, GPU, or TPU support available)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/mkl4r/AIF-Pointing.git
   cd AIF-Pointing
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

   For GPU support:
   ```bash
   pip install -e .[gpu]
   ```

   For TPU support:
   ```bash
   pip install -e .[tpu]
   ```

   For CPU-only (default):
   ```bash
   pip install -e .[cpu]
   ```

## Repository Structure

### Core Components

- **`run_simulation.py`**: Contains the mouse pointing environment implementation. Defines the Active Inference agent and runs simulations. Uses the AIF agent from [difai-base](https://github.com/mkl4r/difai-base) (our package for general Active Inference agents). Results are saved to `data/sim_data/`
- **`plotting.py`**: Plotting utilities and visualization routines
- **`create_plots.ipynb`**: Jupyter notebook for generating plots similar to those in the paper
- **`pyscript_data.py`**: Data loading utilities for user experiment data from `data/user_data/`

### Data Structure

- **`data/sim_data/`**: Simulation results and outputs
- **`data/user_data/`**: Real user experiment data (P1-P6 log files)
- **`data/plots/`**: Generated plot outputs

## Usage

### Running Simulations
To run simulations with the Active Inference agent:
```bash
python run_simulation.py
```

You can modify agent parameters using the `set_params_with_defaults` method in the script.

### Creating Plots
Use the Jupyter notebook to generate visualizations:
```bash
jupyter notebook create_plots.ipynb
```

This notebook creates plots similar to those presented in the paper.

### Loading User Data
Use `pyscript_data.py` to load and process real user experiment data from the `data/user_data/` directory.

## Troubleshooting

### CUDA Version Compatibility
The current setup uses JAX for CUDA 13 (keep in mind that GPU support is very limited on Windows/Mac). If you have CUDA 12.x installed, manually install the appropriate JAX version before installing aif-pointing:

```bash
# For CUDA 12.x, install JAX manually first:
pip install --upgrade "jax[cuda12]" 

# Then install aif-pointing:
pip install -e .
```

Check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the correct JAX version for your CUDA setup.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{klar2025aif,
  title={An Active Inference Model of Mouse Point-and-Click Behaviour},
  author={Markus Klar and Sebastian Stein and Fraser Paterson and John H. Williamson and Roderick Murray-Smith},
  booktitle={6th International Workshop on Active Inference},
  year={2025},
  url={https://arxiv.org/abs/2510.14611}
}
```

## Contact

- **Author**: Markus Klar
- **Website**: [mkl4r.github.io](mkl4r.github.io)
- **Email**: markus.klar@glasgow.ac.uk