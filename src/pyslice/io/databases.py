"""Search crystal-structure databases and retrieve CIF files for PySlice.

This module queries the Materials Project (MP) and the Crystallography Open
Database (COD) over their public REST interfaces and downloads matching
structures as CIF files, which feed directly into
:class:`pyslice.io.loader.Loader`. Only the Python standard library is used,
so no extra dependencies are required. Materials Project requires a free API
key (https://next-gen.materialsproject.org/api) supplied via the ``api_key``
argument or the ``PYSLICE_MP_API_KEY`` / ``MP_API_KEY`` environment
variables; COD is keyless.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

MP_API_BASE = "https://api.materialsproject.org"
COD_BASE = "https://www.crystallography.net/cod"
_MP_KEY_ENV_VARS = ("PYSLICE_MP_API_KEY", "MP_API_KEY")
_USER_AGENT = "pyslice-databases (https://github.com/h-walk/PySlice)"

Provider = Literal["mp", "cod"]


class DatabaseError(RuntimeError):
    """Raised when a structure-database request cannot be completed.

    Wraps network failures, authentication problems, and malformed responses
    from the Materials Project or COD REST APIs with an actionable message.

    Parameters
    ----------
    message : str
        Human-readable description including the suggested fix.

    Attributes
    ----------
    args : tuple
        Standard exception arguments; ``args[0]`` is the message.

    Methods
    -------
    (inherits all behavior from RuntimeError)

    Raises
    ------
    (never raises on construction)

    See Also
    --------
    search_structures : Query a database for matching entries.
    fetch_cif : Download one entry as a CIF file.

    Notes
    -----
    A dedicated type lets callers (e.g. the MCP service layer) distinguish
    database problems from programming errors.

    Examples
    --------
    >>> raise DatabaseError("COD returned no entries for formula 'Xx'")
    Traceback (most recent call last):
    ...
    pyslice.io.databases.DatabaseError: COD returned no entries for formula 'Xx'
    """


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> bytes:
    """Perform an HTTP GET request and return the raw response body.

    Uses :mod:`urllib.request`, which honors standard proxy environment
    variables (``HTTPS_PROXY`` etc.) automatically.

    Parameters
    ----------
    url : str
        Fully encoded URL to request.
    headers : dict[str, str] | None, optional
        Extra request headers (e.g. ``{"X-API-KEY": ...}``).
    timeout : float, optional
        Socket timeout in seconds, by default 30.

    Returns
    -------
    bytes
        The response body.

    Raises
    ------
    DatabaseError
        If the request fails at the HTTP or network level; the message names
        the status code or underlying reason.

    See Also
    --------
    _http_get_json : Same request decoded as JSON.

    Notes
    -----
    401/403 responses are reported as authentication problems so agents know
    to check the API key rather than retry.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise DatabaseError(
                f"Request to {url.split('?')[0]} was rejected ({exc.code}). "
                "Check the API key (Materials Project keys come from "
                "https://next-gen.materialsproject.org/api and are read from "
                "PYSLICE_MP_API_KEY or MP_API_KEY)."
            ) from exc
        if exc.code == 404:
            raise DatabaseError(f"Not found: {url.split('?')[0]}. Check the entry id.") from exc
        raise DatabaseError(f"HTTP {exc.code} from {url.split('?')[0]}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DatabaseError(f"Could not reach {url.split('?')[0]}: {exc.reason}") from exc


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> Any:
    """Perform an HTTP GET request and decode the JSON response.

    Thin wrapper over :func:`_http_get` that adds JSON decoding with a
    consistent error type.

    Parameters
    ----------
    url : str
        Fully encoded URL to request.
    headers : dict[str, str] | None, optional
        Extra request headers.
    timeout : float, optional
        Socket timeout in seconds, by default 30.

    Returns
    -------
    Any
        The decoded JSON payload (dict or list).

    Raises
    ------
    DatabaseError
        If the request fails or the body is not valid JSON.

    See Also
    --------
    _http_get : Raw-bytes variant.
    """
    body = _http_get(url, headers=headers, timeout=timeout)
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabaseError(f"Response from {url.split('?')[0]} is not valid JSON: {exc}") from exc


def _resolve_mp_api_key(api_key: Optional[str]) -> str:
    """Return a usable Materials Project API key or raise with instructions.

    Checks the explicit argument first, then the ``PYSLICE_MP_API_KEY`` and
    ``MP_API_KEY`` environment variables.

    Parameters
    ----------
    api_key : str | None
        Key passed by the caller, if any.

    Returns
    -------
    str
        The resolved API key.

    Raises
    ------
    DatabaseError
        If no key is available; the message says where to obtain one.

    See Also
    --------
    search_structures : Uses this for ``provider="mp"``.
    """
    if api_key:
        return api_key
    for var in _MP_KEY_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    raise DatabaseError(
        "A Materials Project API key is required. Get a free key at "
        "https://next-gen.materialsproject.org/api and pass it as api_key= "
        "or set PYSLICE_MP_API_KEY (or MP_API_KEY). "
        "Alternatively use provider='cod', which needs no key."
    )


def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula string into an element→count mapping.

    Handles simple element+count formulas such as ``"SiO2"`` or ``"Ba Ti O3"``;
    parentheses are not supported.

    Parameters
    ----------
    formula : str
        Formula such as ``"SiO2"``, ``"BN"``, or ``"H2 O"``.

    Returns
    -------
    dict[str, int]
        Mapping of element symbol to (integer) count, insertion-ordered.

    Raises
    ------
    ValueError
        If the string contains anything other than element symbols and
        integer counts.

    See Also
    --------
    _hill_formula : Renders the mapping in Hill order for COD queries.

    Examples
    --------
    >>> _parse_formula("SiO2")
    {'Si': 1, 'O': 2}
    """
    cleaned = formula.replace(" ", "")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", cleaned)
    consumed = "".join(symbol + count for symbol, count in tokens)
    if not cleaned or consumed != cleaned:
        raise ValueError(
            f"Cannot parse formula {formula!r}; expected element symbols with "
            "optional integer counts, e.g. 'SiO2' or 'BaTiO3'."
        )
    counts: Dict[str, int] = {}
    for symbol, count in tokens:
        counts[symbol] = counts.get(symbol, 0) + (int(count) if count else 1)
    return counts


def _hill_formula(formula: str) -> str:
    """Convert a formula to Hill notation with spaces, as COD stores it.

    Hill order puts carbon first and hydrogen second when carbon is present;
    all remaining elements (or all elements, without carbon) are alphabetical.
    Counts of one are omitted, matching COD's ``formula`` search field.

    Parameters
    ----------
    formula : str
        Formula such as ``"SiO2"`` or ``"C2H6O"``.

    Returns
    -------
    str
        Space-separated Hill formula, e.g. ``"O2 Si"`` or ``"C2 H6 O"``.

    Raises
    ------
    ValueError
        If the formula cannot be parsed.

    See Also
    --------
    search_structures : Uses this for COD formula queries.

    Examples
    --------
    >>> _hill_formula("SiO2")
    'O2 Si'
    """
    counts = _parse_formula(formula)
    ordered: List[str] = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
        ordered.extend(sorted(k for k in counts if k not in ("C", "H")))
    else:
        ordered.extend(sorted(counts))
    return " ".join(f"{el}{counts[el] if counts[el] != 1 else ''}" for el in ordered)


def _search_mp(
    formula: Optional[str],
    elements: Optional[Sequence[str]],
    limit: int,
    api_key: Optional[str],
    timeout: float,
) -> List[Dict[str, Any]]:
    """Query the Materials Project summary endpoint.

    Builds a ``/materials/summary/`` request filtered by formula and/or
    element set and normalizes each hit into the provider-independent entry
    format used by :func:`search_structures`.

    Parameters
    ----------
    formula : str | None
        Chemical formula filter (e.g. ``"SiO2"``); MP also accepts wildcards
        such as ``"Si*"``.
    elements : Sequence[str] | None
        Elements that must all be present (e.g. ``("Ga", "N")``).
    limit : int
        Maximum number of entries to return.
    api_key : str | None
        Materials Project API key; resolved via :func:`_resolve_mp_api_key`.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    list[dict]
        Normalized entries sorted by ``energy_above_hull`` (most stable
        first) when that field is present.

    Raises
    ------
    DatabaseError
        On missing key, network failure, or malformed response.

    See Also
    --------
    search_structures : Public entry point dispatching to this.
    """
    key = _resolve_mp_api_key(api_key)
    params: Dict[str, str] = {
        "_fields": "material_id,formula_pretty,symmetry,nsites,energy_above_hull,volume,density",
        "_limit": str(limit),
    }
    if formula:
        params["formula"] = formula
    if elements:
        params["elements"] = ",".join(elements)
    url = f"{MP_API_BASE}/materials/summary/?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, headers={"X-API-KEY": key}, timeout=timeout)
    entries: List[Dict[str, Any]] = []
    for item in payload.get("data", []):
        symmetry = item.get("symmetry") or {}
        entries.append({
            "provider": "mp",
            "id": item.get("material_id"),
            "formula": item.get("formula_pretty"),
            "spacegroup": symmetry.get("symbol"),
            "crystal_system": symmetry.get("crystal_system"),
            "nsites": item.get("nsites"),
            "energy_above_hull_eV": item.get("energy_above_hull"),
            "volume_A3": item.get("volume"),
            "density_g_cm3": item.get("density"),
        })
    entries.sort(key=lambda e: (e["energy_above_hull_eV"] is None, e["energy_above_hull_eV"] or 0.0))
    return entries


def _search_cod(
    formula: Optional[str],
    elements: Optional[Sequence[str]],
    limit: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Query the COD ``result`` endpoint.

    Filters by Hill formula and/or a strict element set and normalizes each
    hit into the provider-independent entry format used by
    :func:`search_structures`.

    Parameters
    ----------
    formula : str | None
        Chemical formula; converted to spaced Hill notation for COD.
    elements : Sequence[str] | None
        Elements that must all be present; COD is additionally told to
        exclude entries with other elements (strict match).
    limit : int
        Maximum number of entries to return (applied client-side; COD has no
        limit parameter).
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    list[dict]
        Normalized entries.

    Raises
    ------
    DatabaseError
        On network failure or malformed response.
    ValueError
        If more than eight elements are given (COD supports el1..el8).

    See Also
    --------
    search_structures : Public entry point dispatching to this.
    """
    params: Dict[str, str] = {"format": "json"}
    if formula:
        params["formula"] = _hill_formula(formula)
    if elements:
        elements = list(elements)
        if len(elements) > 8:
            raise ValueError("COD element search supports at most 8 elements (el1..el8).")
        for i, element in enumerate(elements, start=1):
            params[f"el{i}"] = element
        params["strictmin"] = str(len(elements))
        params["strictmax"] = str(len(elements))
    url = f"{COD_BASE}/result?{urllib.parse.urlencode(params)}"
    payload = _http_get_json(url, timeout=timeout)
    if not isinstance(payload, list):
        raise DatabaseError(f"Unexpected COD response type {type(payload).__name__}; expected a list.")
    entries: List[Dict[str, Any]] = []
    for item in payload[:limit]:
        entries.append({
            "provider": "cod",
            "id": str(item.get("file")),
            "formula": (item.get("formula") or "").strip("- "),
            "spacegroup": item.get("sg"),
            "cell_A": [item.get("a"), item.get("b"), item.get("c")],
            "cell_angles_deg": [item.get("alpha"), item.get("beta"), item.get("gamma")],
            "volume_A3": item.get("vol"),
            "title": item.get("title"),
            "journal": item.get("journal"),
            "year": item.get("year"),
        })
    return entries


def search_structures(
    provider: Provider,
    formula: Optional[str] = None,
    elements: Optional[Sequence[str]] = None,
    limit: int = 20,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Search a crystal-structure database for matching entries.

    Queries the Materials Project (``provider="mp"``, API key required) or
    the Crystallography Open Database (``provider="cod"``, keyless) by
    chemical formula and/or element set, returning lightweight entry
    summaries whose ``id`` values feed :func:`fetch_cif`.

    Parameters
    ----------
    provider : {"mp", "cod"}
        Database to query.
    formula : str | None, optional
        Chemical formula, e.g. ``"SiO2"`` or ``"BaTiO3"``. For COD the
        formula is converted to Hill notation of the formula unit; for MP it
        is passed through (MP accepts wildcards such as ``"Si*"``).
    elements : Sequence[str] | None, optional
        Element symbols that must all be present, e.g. ``("Ga", "N")``. For
        COD the match is strict (no other elements).
    limit : int, optional
        Maximum number of entries to return, by default 20.
    api_key : str | None, optional
        Materials Project API key; falls back to ``PYSLICE_MP_API_KEY`` /
        ``MP_API_KEY`` environment variables. Ignored for COD.
    timeout : float, optional
        Request timeout in seconds, by default 30.

    Returns
    -------
    list[dict]
        One dict per entry with at least ``provider``, ``id``, ``formula``,
        and ``spacegroup``; MP adds ``energy_above_hull_eV`` (sorted most
        stable first), COD adds cell parameters and publication info.

    Raises
    ------
    ValueError
        If neither ``formula`` nor ``elements`` is given, or the provider is
        unknown.
    DatabaseError
        If the request fails (network, authentication, malformed response).

    See Also
    --------
    fetch_cif : Download one of the returned entries as a CIF file.
    load_structure_from_database : Fetch and load in one step.

    Notes
    -----
    MP entries are computed (DFT-relaxed) structures; COD entries are
    experimental crystal structures from the literature. For simulations of
    a known material either works; prefer MP when you want the ground-state
    computed cell and COD when you want a specific published refinement.

    Examples
    --------
    >>> entries = search_structures("cod", formula="C")  # doctest: +SKIP
    >>> entries[0]["id"]  # doctest: +SKIP
    '9008569'

    References
    ----------
    Materials Project API documentation:
    https://api.materialsproject.org/docs

    COD RESTful API documentation:
    https://wiki.crystallography.net/RESTful_API/
    """
    if not formula and not elements:
        raise ValueError("Provide a formula (e.g. 'SiO2') and/or an elements list (e.g. ['Ga', 'N']).")
    if provider == "mp":
        return _search_mp(formula, elements, limit, api_key, timeout)
    if provider == "cod":
        return _search_cod(formula, elements, limit, timeout)
    raise ValueError(f"Unknown provider {provider!r}; expected 'mp' or 'cod'.")


def _structure_dict_to_cif(structure: Dict[str, Any], data_name: str) -> str:
    """Render a pymatgen-style structure dict as a P1 CIF string.

    The Materials Project API returns structures as pymatgen ``Structure``
    JSON (lattice matrix plus sites with fractional coordinates). This
    renders that dict as a symmetry-free (P1) CIF that ASE — and therefore
    :class:`pyslice.io.loader.Loader` — reads directly, avoiding a pymatgen
    dependency.

    Parameters
    ----------
    structure : dict
        Pymatgen ``Structure.as_dict()`` payload with ``lattice`` and
        ``sites`` keys.
    data_name : str
        CIF data-block name (typically the material id).

    Returns
    -------
    str
        Complete CIF file content.

    Raises
    ------
    DatabaseError
        If the dict lacks lattice parameters or sites.

    See Also
    --------
    fetch_cif : Uses this for Materials Project downloads.

    Notes
    -----
    All sites are written explicitly, so no symmetry information is lost by
    using P1. For partially occupied sites every species is written with its
    occupancy; ordering such sites is the caller's responsibility before
    running a simulation.
    """
    lattice = structure.get("lattice") or {}
    sites = structure.get("sites") or []
    required = ("a", "b", "c", "alpha", "beta", "gamma")
    if not sites or any(lattice.get(k) is None for k in required):
        raise DatabaseError("Materials Project structure payload is missing lattice parameters or sites.")

    lines = [
        f"data_{data_name}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a     {lattice['a']:.6f}",
        f"_cell_length_b     {lattice['b']:.6f}",
        f"_cell_length_c     {lattice['c']:.6f}",
        f"_cell_angle_alpha  {lattice['alpha']:.6f}",
        f"_cell_angle_beta   {lattice['beta']:.6f}",
        f"_cell_angle_gamma  {lattice['gamma']:.6f}",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    label_counts: Dict[str, int] = {}
    for site in sites:
        abc = site.get("abc")
        species = site.get("species") or []
        if abc is None or not species:
            raise DatabaseError("Materials Project site payload is missing coordinates or species.")
        for spec in species:
            element = spec.get("element")
            occupancy = float(spec.get("occu", 1.0))
            label_counts[element] = label_counts.get(element, 0) + 1
            label = f"{element}{label_counts[element]}"
            lines.append(
                f"{element} {label} {abc[0]:.6f} {abc[1]:.6f} {abc[2]:.6f} {occupancy:.4f}"
            )
    return "\n".join(lines) + "\n"


def fetch_cif(
    provider: Provider,
    entry_id: str,
    output_dir: str | Path = ".",
    filename: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Path:
    """Download one database entry as a CIF file.

    For COD the published CIF is downloaded verbatim; for the Materials
    Project the structure JSON is fetched and rendered as a P1 CIF (all
    sites explicit). The resulting file loads with
    ``Loader(filename=str(path)).load()``.

    Parameters
    ----------
    provider : {"mp", "cod"}
        Database the entry comes from.
    entry_id : str
        Entry identifier, e.g. ``"mp-149"`` (MP) or ``"1010939"`` (COD), as
        returned by :func:`search_structures`.
    output_dir : str | pathlib.Path, optional
        Directory to write the CIF into (created if missing), by default the
        current directory.
    filename : str | None, optional
        Output filename; defaults to ``<provider>_<entry_id>.cif``.
    api_key : str | None, optional
        Materials Project API key; falls back to ``PYSLICE_MP_API_KEY`` /
        ``MP_API_KEY``. Ignored for COD.
    timeout : float, optional
        Request timeout in seconds, by default 60.

    Returns
    -------
    pathlib.Path
        Path of the written CIF file.

    Raises
    ------
    ValueError
        If the provider is unknown.
    DatabaseError
        If the entry cannot be retrieved or its payload is malformed.

    See Also
    --------
    search_structures : Find entry ids to fetch.
    load_structure_from_database : Fetch and load in one step.

    Examples
    --------
    >>> path = fetch_cif("cod", "1010939", output_dir="structures")  # doctest: +SKIP
    >>> path.name  # doctest: +SKIP
    'cod_1010939.cif'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entry_id = str(entry_id).strip()

    if provider == "mp":
        key = _resolve_mp_api_key(api_key)
        params = urllib.parse.urlencode({"material_ids": entry_id, "_fields": "material_id,structure"})
        payload = _http_get_json(
            f"{MP_API_BASE}/materials/summary/?{params}",
            headers={"X-API-KEY": key},
            timeout=timeout,
        )
        data = payload.get("data") or []
        if not data or not data[0].get("structure"):
            raise DatabaseError(
                f"Materials Project returned no structure for {entry_id!r}. "
                "Check the id with search_structures(provider='mp', ...)."
            )
        cif_text = _structure_dict_to_cif(data[0]["structure"], entry_id.replace("-", "_"))
    elif provider == "cod":
        if not entry_id.isdigit():
            raise DatabaseError(f"COD ids are numeric (e.g. '1010939'); got {entry_id!r}.")
        cif_text = _http_get(f"{COD_BASE}/{entry_id}.cif", timeout=timeout).decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unknown provider {provider!r}; expected 'mp' or 'cod'.")

    path = output_dir / (filename or f"{provider}_{entry_id.replace('-', '_')}.cif")
    path.write_text(cif_text, encoding="utf-8")
    logger.info("Wrote %s entry %s to %s", provider, entry_id, path)
    return path


def load_structure_from_database(
    provider: Provider,
    entry_id: str,
    output_dir: str | Path = ".",
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    **loader_kwargs: Any,
):
    """Fetch a database entry and load it as a PySlice ``Trajectory``.

    Convenience wrapper chaining :func:`fetch_cif` and
    :class:`pyslice.io.loader.Loader`, so a simulation can start from a
    database id in one call.

    Parameters
    ----------
    provider : {"mp", "cod"}
        Database the entry comes from.
    entry_id : str
        Entry identifier, e.g. ``"mp-149"`` or ``"1010939"``.
    output_dir : str | pathlib.Path, optional
        Directory the CIF (and Loader ``.npy`` cache) is written into.
    api_key : str | None, optional
        Materials Project API key; falls back to the environment variables.
    timeout : float, optional
        Request timeout in seconds, by default 60.
    **loader_kwargs : Any
        Extra keyword arguments forwarded to
        :class:`pyslice.io.loader.Loader` (e.g. ``timestep=``).

    Returns
    -------
    pyslice.multislice.trajectory.Trajectory
        Single-frame trajectory of the retrieved structure with
        element-symbol ``atom_types``.

    Raises
    ------
    ValueError
        If the provider is unknown.
    DatabaseError
        If the entry cannot be retrieved.

    See Also
    --------
    search_structures : Find entry ids.
    fetch_cif : Just download the CIF without loading.

    Examples
    --------
    >>> traj = load_structure_from_database("cod", "1010939")  # doctest: +SKIP
    >>> traj.n_atoms  # doctest: +SKIP
    8
    """
    from .loader import Loader

    cif_path = fetch_cif(provider, entry_id, output_dir=output_dir, api_key=api_key, timeout=timeout)
    return Loader(filename=str(cif_path), **loader_kwargs).load()
