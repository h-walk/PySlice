"""Lightweight checks for documentation examples and repository onboarding."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TUTORIAL = ROOT / "example.ipynb"
EXAMPLE_INDEX = ROOT / "examples" / "README.md"

USER_GUIDES = (
    "loading-trajectories.md",
    "multislice-grids.md",
    "stem-and-4dstem.md",
    "tacaw-analysis.md",
    "md-to-tacaw.md",
    "results-and-storage.md",
    "scaling.md",
    "troubleshooting.md",
)

TRACKED_EXAMPLES = (
    "aberrations.py",
    "haadf_stem.py",
    "k_space_tmdc_showcase_pub.py",
    "lacbed.py",
    "loading_trajectories.py",
    "molecular_dynamics.py",
    "real_space_phonon_showcase_pub.py",
    "tacaw_from_trajectory.py",
    "tacaw_pipeline.py",
    "tacaw_spectrum_image.py",
    "tem_diffraction.py",
)


def test_readme_python_blocks_compile():
    """Keep every Python snippet in the canonical README syntactically valid."""
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert blocks, "README.md contains no Python examples"
    for index, source in enumerate(blocks, start=1):
        compile(source, f"README.md python block {index}", "exec")


def test_user_guide_python_blocks_compile():
    """Compile every advertised Python snippet, not only the README example."""
    for name in USER_GUIDES:
        path = ROOT / "docs" / "user-guide" / name
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.DOTALL)
        for index, source in enumerate(blocks, start=1):
            compile(source, f"{path.name} python block {index}", "exec")


def test_documented_python_version_matches_package_metadata():
    """Prevent the README from advertising an interpreter pip will reject."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.12"
    assert "Python 3.12 or newer" in README.read_text()


def test_readme_editable_install_starts_from_a_checkout():
    """Do not advertise an editable install before obtaining the repository."""
    readme = README.read_text()
    clone_index = readme.index("git clone https://github.com/h-walk/PySlice.git")
    cd_index = readme.index("cd PySlice", clone_index)
    install_index = readme.index('python -m pip install -e ".[fast]"', cd_index)
    assert clone_index < cd_index < install_index


def test_tracked_examples_compile_and_do_not_escape_the_repository():
    """Examples documented as root-runnable must use repository-local inputs."""
    for name in TRACKED_EXAMPLES:
        path = ROOT / "examples" / name
        source = path.read_text()
        ast.parse(source, filename=str(path))
        assert "../tests/inputs" not in source


def test_examples_use_cycles_per_angstrom_reciprocal_paths():
    """Prevent accidental angular-wavevector factors in TACAW paths."""
    pipeline = (ROOT / "examples" / "tacaw_pipeline.py").read_text()
    loaded = (ROOT / "examples" / "tacaw_from_trajectory.py").read_text()
    assert "2 * np.pi / a_si" not in pipeline
    assert "1 / (3 * a_hbn)" not in loaded
    assert "np.linspace(-1 / a_si, 1 / a_si" in pipeline
    assert "reciprocal_lattice_xy = np.linalg.inv(direct_lattice_xy).T" in loaded
    assert "G_x = reciprocal_lattice_xy[0, 0]" in loaded


def test_user_guides_and_example_index_exist_and_are_linked():
    readme = README.read_text()
    assert EXAMPLE_INDEX.exists()
    assert "examples/README.md" in readme
    for guide in USER_GUIDES:
        path = ROOT / "docs" / "user-guide" / guide
        assert path.exists(), f"missing user guide: {guide}"
        assert f"docs/user-guide/{guide}" in readme


def test_example_index_discloses_cache_writes_outside_outputs():
    example_index = EXAMPLE_INDEX.read_text()
    assert "psi_data/" in example_index
    assert ".cache.json" in example_index
    assert "tests/inputs/" in example_index


def test_public_haadf_examples_use_on_the_fly_detector_integration():
    for path in (
        ROOT / "examples" / "haadf_stem.py",
        ROOT / "examples" / "aberrations.py",
    ):
        source = path.read_text()
        assert "ADF=(60, 200)" in source
        assert "return_layers=None" in source
        assert "cache_wavefunctions=False" in source
        assert "fold_positions_to_orthogonal_box" in source

    readme = README.read_text()
    assert "ADF=(inner, outer)" in readme
    assert "return_layers=None" in readme
    assert "cache_wavefunctions=False" in readme


def test_tilted_hbn_examples_fold_before_spatial_use():
    """Canonical examples must honor the orthogonal-cell simulation contract."""
    for name, later_operation in (
        ("tacaw_from_trajectory.py", "calc.setup("),
        ("haadf_stem.py", "trajectory_md.slice_positions("),
        ("loading_trajectories.py", "cropped = traj.slice_positions("),
    ):
        source = (ROOT / "examples" / name).read_text()
        fold_index = source.rindex("fold_positions_to_orthogonal_box()", 0, source.index(later_operation))
        assert fold_index < source.index(later_operation)


def test_examples_report_tacaw_frequency_and_reload_timestep_safely():
    loaded = (ROOT / "examples" / "tacaw_from_trajectory.py").read_text()
    image = (ROOT / "examples" / "tacaw_spectrum_image.py").read_text()
    assert "nearest_frequency" in loaded
    assert "Frequency range: {freqs[0]" in loaded
    assert "timestep=0.025" in image
    assert "spectrum_image_reshaped" in image


def test_tutorial_notebook_structure_and_code():
    """Validate the tracked tutorial without executing expensive simulations."""
    notebook = json.loads(TUTORIAL.read_text())
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, "example.ipynb contains no executable examples"
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        compile(source, f"example.ipynb code cell {index}", "exec")
