# Installation Instructions for TACAW

## Prerequisites

- Python 3.12 or higher
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

   - **abtem**: The abtem library may require specific installation steps. Check the [abtem documentation](https://abtem.readthedocs.io/en/latest/) for detailed installation instructions.

## Testing the Installation

Run the main simulation script to verify everything is working:
```bash
python3 main.py
```

## Troubleshooting

- **OVITO installation issues**: OVITO sometimes requires specific Python versions. Check the [OVITO documentation](https://www.ovito.org/docs/current/python/) for compatibility.

- **abtem installation**: If you encounter abtem issues, refer to the [abtem installation guide](https://abtem.readthedocs.io/en/latest/installation.html).

- **Memory issues**: For large trajectories, you may need to adjust the `batch_size` parameter or limit the number of frames processed.

- **ASE compatibility**: Ensure your ASE version is compatible with abtem. The requirements.txt specifies compatible versions.
