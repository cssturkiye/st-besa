# ST-BESA: Spatio-Temporal Built Environment & Settlement Analysis Platform
![ST-BESA Banner](img/st-besa-banner.png)



<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cssturkiye/st-besa/blob/main/ST_BESA_Colab.ipynb)
[![Paper: Under Review](https://img.shields.io/badge/Paper-Under%20Review-orange)](#-about-the-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

*ST-BESA: A low-code platform for spatio-temporal settlement analytics*

</div>

> We will update this README with a DOI link once available.

## Overview

ST-BESA is a platform for computing built-environment indicators and settlement classification metrics for administrative units. It integrates Google Earth Engine with JRC Global Human Settlement Layer (GHSL) datasets to provide:

- **Building Volume** (m³)
- **Building Surface** (m²)
- **Population Count**
- **BVPC** (Building Volume Per Capita)
- **BSPC** (Building Surface Per Capita)
- **SMOD Classification** (Degree of Urbanization L1/L2)

![ST-BESA Demo](img/ST-BESA.gif)

## Features

| Feature | Description |
|---------|-------------|
| Multi-year analysis | Compute indicators for years 1975–2030 (5-year intervals) |
| Instant year switching | Switch between years without re-computation |
| Auto/Manual scaling | Percentile-based or user-defined visualization ranges |
| High-resolution export | 600 DPI map layers for publication |
| Excel reports | Multi-sheet workbooks with data dictionary |

## Quick Start

### Local Installation

```bash
git clone https://github.com/cssturkiye/st-besa.git
cd st-besa
pip install -r requirements.txt
python app.py
```

### Google Colab

Click the **Open in Colab** badge above to run ST-BESA in the cloud without local installation.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Setup instructions and authentication |
| [User Guide](docs/USER_GUIDE.md) | How to use the application |
| [Architecture](docs/ARCHITECTURE.md) | Technical design and module structure |
| [Boundary Configuration](docs/BOUNDARY_CONFIG.md) | Adding new countries/datasets |

## Data Sources

| Dataset | Provider | License |
|---------|----------|---------|
| Administrative Boundaries | UN OCHA / Kontur | CC BY 4.0 |
| Building Volume/Surface | JRC GHSL (GHS_BUILT_V/S) | CC BY 4.0 |
| Population | JRC GHSL (GHS_POP) | CC BY 4.0 |
| Settlement Model | JRC GHSL (GHS_SMOD) | CC BY 4.0 |

## Requirements

- Python 3.10+
- Google Cloud Project with Earth Engine API enabled
- See [requirements.txt](requirements.txt) for dependencies

## Citation

If you use ST-BESA in your research, please cite:

```bibtex
@article{stbesa2026,
  title={ST-BESA: A Platform for Spatio-Temporal Built Environment and Settlement Analysis},
  author={<Authors>},
  journal={SoftwareX},
  year={2026},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Data sources are subject to their respective licenses (CC BY 4.0 for JRC GHSL and OCHA datasets).
