# Design-Friendly Wind Farm Control

Yaw control setpoint estimation for wind farms using layout-agnostic Graph Neural Networks (GNNs).

## Installation

```bash
pip install torch torch-geometric
pip install -e .
```

## Usage

```python
from design_friendly.utils.easy import easy_yaw_gnn

# Predict yaw angles
yaw = easy_yaw_gnn(x, y, wd, ws, TI)
```

See `easy.ipynb` for examples.

## Citation

If you use this code or the models, please cite:

> Dirik, D. G., Réthoré, P.-E., Riva, R., & Quick, J. (2026). Design-friendly wind farm control setpoint estimation via layout-agnostic graph neural networks. *Journal of Physics: Conference Series*.

```bibtex
@article{dirik_design-friendly_2026,
    title   = {Design-friendly wind farm control setpoint estimation via layout-agnostic graph neural networks},
    journal = {Journal of Physics: Conference Series},
    author  = {Dirik, Deniz Gokhan and Réthoré, Pierre-Elouan and Riva, Riccardo and Quick, Julian},
    year    = {2026},
}
```
