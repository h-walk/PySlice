# Installation Instructions for JACR TACAW

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv tacaw_env
   source tacaw_env/bin/activate  # On Windows: tacaw_env\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Special installation notes:**

   - **OVITO**: If you encounter issues installing OVITO via pip, you may need to install it separately:
     ```bash
     pip install ovito --find-links https://www.ovito.org/pip/
     ```

   - **ptyrodactyl**: If ptyrodactyl is not available via pip, you may need to install from source:
     ```bash
     pip install git+https://github.com/PyTEM/ptyrodactyl.git
     ```

   - **JAX with GPU support** (optional, for faster computation):
     ```bash
     pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```

## Testing the Installation

Run the test script to verify everything is working:
```bash
python3 test_jacr_simulation.py
```

## Troubleshooting

- **OVITO installation issues**: OVITO sometimes requires specific Python versions. Check the [OVITO documentation](https://www.ovito.org/docs/current/python/) for compatibility.

- **ptyrodactyl not found**: This package may need to be installed from source or a specific repository. Check the ptyrodactyl documentation for installation instructions.

- **JAX installation**: If you encounter JAX issues, refer to the [JAX installation guide](https://github.com/google/jax#installation).

- **Memory issues**: For large trajectories, you may need to adjust the `supersampling` parameter or limit the number of frames processed.
