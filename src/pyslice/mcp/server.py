"""MCP server implementation for PySlice.

Exposes PySlice's simulation pipeline — structure-database search, structure
loading and building, physics-based parameter advice, MD, multislice,
HAADF/TACAW post-processing, and ``.sea`` export — as ``pyslice_*`` MCP
tools. Every tool is a thin wrapper over :class:`~pyslice.mcp.service.PySliceService`,
mirroring the ``pySEA.sea_eco.mcp`` layout.
"""
from __future__ import annotations

from argparse import ArgumentParser

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp>=2.0 renamed FastMCP to MCPServer with the same tool API
    from mcp.server import MCPServer as FastMCP

from pyslice.mcp.service import (
    BuildSlabInput,
    ComputeHAADFInput,
    ComputeTACAWInput,
    DispersionInput,
    ExportSeaFileInput,
    ExportSeaInput,
    FetchStructureInput,
    HandleInput,
    LoadStructureInput,
    PlanSimulationInput,
    PreviewPotentialInput,
    PySliceService,
    RenderSignalInput,
    ResponseFormat,
    RunMDInput,
    RunMultisliceInput,
    SearchStructuresInput,
    SetupMultisliceInput,
    SpectrumImageInput,
    SuggestParametersInput,
    TACAWSpectrumInput,
    TransformTrajectoryInput,
)


def _tool_annotations(read_only: bool, open_world: bool = False) -> dict[str, bool | str]:
    """Build standard MCP tool annotations.

    Parameters
    ----------
    read_only : bool
        Whether the tool leaves the environment unchanged.
    open_world : bool, optional
        Whether the tool talks to external services (databases), by default
        False.

    Returns
    -------
    dict[str, bool | str]
        Annotation mapping for the ``@mcp.tool`` decorator.

    See Also
    --------
    build_server : Applies these to every tool.
    """
    return {
        "title": "PySlice MCP Tool",
        "readOnlyHint": read_only,
        "destructiveHint": False,
        "idempotentHint": read_only,
        "openWorldHint": open_world,
    }


def build_server(
    workspace: str | None = None,
    service: PySliceService | None = None,
) -> FastMCP:
    """Create a configured FastMCP server for PySlice.

    Parameters
    ----------
    workspace : str | None, optional
        Artifact directory; defaults to ``PYSLICE_MCP_WORKSPACE`` or the
        current directory.
    service : PySliceService | None, optional
        Pre-built service (useful in tests); a new one is created otherwise.

    Returns
    -------
    mcp.server.fastmcp.FastMCP
        Server with every ``pyslice_*`` tool registered.

    See Also
    --------
    pyslice.mcp.service.PySliceService : Implements the tool logic.

    Examples
    --------
    >>> server = build_server(workspace="/tmp/pyslice-mcp")  # doctest: +SKIP
    >>> server.run()  # doctest: +SKIP
    """
    service = service or PySliceService(workspace=workspace)
    mcp = FastMCP(
        "pyslice_mcp",
        instructions=(
            "PySlice runs multislice electron-scattering simulations (TEM/STEM/4D-STEM imaging, "
            "diffraction, and TACAW vibrational-EELS from MD) and exposes them through MCP. "
            "Call `pyslice_get_conventions` first in zero-context sessions — it documents units "
            "(Angstroms, mrad, eV, ps/fs, THz), the canonical workflow, and PySlice's gotchas. "
            "Structures come from local files (`pyslice_load_structure`) or from the Materials "
            "Project / COD (`pyslice_search_structures` + `pyslice_fetch_structure`). Most tools "
            "return handles referring to live objects in the session registry; persist results "
            "with `pyslice_export_sea`. Use `pyslice_suggest_parameters` before setting up a "
            "simulation — it encodes the sampling/slice/probe/frame physics."
        ),
    )

    @mcp.tool(name="pyslice_get_conventions", annotations=_tool_annotations(True))
    async def pyslice_get_conventions(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
        """Return PySlice MCP conventions: units, workflow, gotchas, examples. Call this first."""

        return service.format_response(service.get_conventions(), response_format)

    @mcp.tool(name="pyslice_get_workspace", annotations=_tool_annotations(True))
    async def pyslice_get_workspace(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
        """Return the artifact workspace path and registry size."""

        return service.format_response(service.get_workspace(), response_format)

    @mcp.tool(name="pyslice_list_handles", annotations=_tool_annotations(True))
    async def pyslice_list_handles(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
        """List registered object handles (trajectories, calculators, results)."""

        return service.format_response(service.list_handles(), response_format)

    @mcp.tool(name="pyslice_describe_handle", annotations=_tool_annotations(True))
    async def pyslice_describe_handle(params: HandleInput) -> str:
        """Describe one registered handle (type-specific summary)."""

        return service.format_response(service.describe_handle(params.handle), params.response_format)

    @mcp.tool(name="pyslice_search_structures", annotations=_tool_annotations(True, open_world=True))
    async def pyslice_search_structures(params: SearchStructuresInput) -> str:
        """Search Materials Project ('mp', API key) or COD ('cod', keyless) for crystal structures by formula/elements."""

        payload = service.search_structures(
            params.provider, params.formula, params.elements, params.limit, params.api_key
        )
        return service.format_response(payload, params.response_format)

    @mcp.tool(name="pyslice_fetch_structure", annotations=_tool_annotations(False, open_world=True))
    async def pyslice_fetch_structure(params: FetchStructureInput) -> str:
        """Download a database entry as a CIF into the workspace and (by default) load it as a Trajectory handle."""

        payload = service.fetch_structure(
            params.provider, params.entry_id, params.filename, params.load, params.timestep_ps, params.api_key
        )
        return service.format_response(payload, params.response_format)

    @mcp.tool(name="pyslice_load_structure", annotations=_tool_annotations(False))
    async def pyslice_load_structure(params: LoadStructureInput) -> str:
        """Load a local CIF/XYZ/LAMMPS/ASE-traj file into a Trajectory handle (pass atom_mapping for LAMMPS type ids)."""

        payload = service.load_structure(params.path, params.atom_mapping, params.timestep_ps)
        return service.format_response(payload, params.response_format)

    @mcp.tool(name="pyslice_transform_trajectory", annotations=_tool_annotations(False))
    async def pyslice_transform_trajectory(params: TransformTrajectoryInput) -> str:
        """Build structures: tile supercells, rotate to a zone axis, tilt, crop, select frames, or add frozen-phonon displacements."""

        payload = service.transform_trajectory(params.handle, params.operations, params.name)
        return service.format_response(payload, params.response_format)

    @mcp.tool(name="pyslice_build_slab", annotations=_tool_annotations(False))
    async def pyslice_build_slab(params: BuildSlabInput) -> str:
        """Build an exactly periodic beam-oriented slab (ASE surface + orthogonalized cell) from a unit-cell handle — no carved edges."""

        return service.format_response(service.build_slab(params), params.response_format)

    @mcp.tool(name="pyslice_suggest_parameters", annotations=_tool_annotations(True))
    async def pyslice_suggest_parameters(params: SuggestParametersInput) -> str:
        """Suggest justified multislice parameters (sampling, slices, tiling, probes, frames) for a goal and structure."""

        return service.format_response(service.suggest_parameters(params), params.response_format)

    @mcp.tool(name="pyslice_plan_simulation", annotations=_tool_annotations(True))
    async def pyslice_plan_simulation(params: PlanSimulationInput) -> str:
        """Turn a structured simulation request into a full confirmable plan: parameter table with supplied/derived/default origins, build + thermal + setup + post-processing plans, and open questions. Present it for confirmation before executing."""

        return service.format_response(service.plan_simulation(params), params.response_format)

    @mcp.tool(name="pyslice_run_md", annotations=_tool_annotations(False))
    async def pyslice_run_md(params: RunMDInput) -> str:
        """Run ML-potential molecular dynamics (ORB/FAIRChem) from a structure handle; blocking, needs the [md] extra."""

        return service.format_response(service.run_md(params), params.response_format)

    @mcp.tool(name="pyslice_setup_multislice", annotations=_tool_annotations(False))
    async def pyslice_setup_multislice(params: SetupMultisliceInput) -> str:
        """Configure a multislice run; returns the calculator handle plus grid size and a memory estimate to check before running."""

        return service.format_response(service.setup_multislice(params), params.response_format)

    @mcp.tool(name="pyslice_run_multislice", annotations=_tool_annotations(False))
    async def pyslice_run_multislice(params: RunMultisliceInput) -> str:
        """Execute a configured multislice calculation (blocking); returns a WFData handle (+ HAADF handle when ADF was set)."""

        payload = service.run_multislice(params.calculator_handle, params.force_rerun)
        return service.format_response(payload, params.response_format)

    @mcp.tool(name="pyslice_compute_haadf", annotations=_tool_annotations(False))
    async def pyslice_compute_haadf(params: ComputeHAADFInput) -> str:
        """Integrate a HAADF/ADF image from wavefunction data with chosen detector angles; optionally render a PNG."""

        return service.format_response(service.compute_haadf(params), params.response_format)

    @mcp.tool(name="pyslice_compute_tacaw", annotations=_tool_annotations(False))
    async def pyslice_compute_tacaw(params: ComputeTACAWInput) -> str:
        """FFT multi-frame wavefunction data over time into TACAW spectral data (optionally Bose-corrected)."""

        return service.format_response(service.compute_tacaw(params), params.response_format)

    @mcp.tool(name="pyslice_tacaw_spectrum", annotations=_tool_annotations(False))
    async def pyslice_tacaw_spectrum(params: TACAWSpectrumInput) -> str:
        """Extract a k-integrated TACAW spectrum and report its dominant phonon peaks in THz."""

        return service.format_response(service.tacaw_spectrum(params), params.response_format)

    @mcp.tool(name="pyslice_spectrum_image", annotations=_tool_annotations(False))
    async def pyslice_spectrum_image(params: SpectrumImageInput) -> str:
        """Map TACAW intensity at one frequency across the probe grid (real-space phonon map); optionally render a PNG."""

        return service.format_response(service.spectrum_image(params), params.response_format)

    @mcp.tool(name="pyslice_dispersion", annotations=_tool_annotations(False))
    async def pyslice_dispersion(params: DispersionInput) -> str:
        """Extract a phonon dispersion (frequency vs k) along a k-path from TACAW data; optionally render a PNG."""

        return service.format_response(service.dispersion(params), params.response_format)

    @mcp.tool(name="pyslice_preview_potential", annotations=_tool_annotations(False))
    async def pyslice_preview_potential(params: PreviewPotentialInput) -> str:
        """Render the projected potential of a structure as a fast sanity check before running multislice."""

        return service.format_response(service.preview_potential(params), params.response_format)

    @mcp.tool(name="pyslice_render_signal", annotations=_tool_annotations(False))
    async def pyslice_render_signal(params: RenderSignalInput) -> str:
        """Render a result handle with sea-eco plotting (calibrated axes) to a PNG or plotly HTML artifact."""

        return service.format_response(service.render_signal(params), params.response_format)

    @mcp.tool(name="pyslice_export_sea", annotations=_tool_annotations(False))
    async def pyslice_export_sea(params: ExportSeaInput) -> str:
        """Export a WFData/HAADFData/TACAWData handle to a calibrated .sea file readable across the pySEA ecosystem."""

        return service.format_response(service.export_sea(params.handle, params.filename), params.response_format)

    @mcp.tool(name="pyslice_export_sea_file", annotations=_tool_annotations(False))
    async def pyslice_export_sea_file(params: ExportSeaFileInput) -> str:
        """Package results + materials provenance into one SEAFile: Simulations plus Materials entries (Material with database info, Sample with Metadata.build, Sample rooted at Material)."""

        return service.format_response(service.export_sea_file(params), params.response_format)

    return mcp


def build_parser() -> ArgumentParser:
    """Build the command-line parser for the MCP server.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``--workspace`` option.
    """
    parser = ArgumentParser(description="Run the PySlice MCP server (stdio).")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Directory for generated artifacts (CIFs, .sea exports, PNGs). Defaults to PYSLICE_MCP_WORKSPACE or the current directory.",
    )
    return parser


def main() -> int:
    """Run the MCP server in stdio mode.

    Returns
    -------
    int
        Process exit code (0 on clean shutdown).
    """
    parser = build_parser()
    args = parser.parse_args()
    server = build_server(workspace=args.workspace)
    server.run()
    return 0
