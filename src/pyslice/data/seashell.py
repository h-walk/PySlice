"""The sea-eco resolution layer — PySlice's only sea-eco coupling point.

Every PySlice result is meant to *be* a sea-eco object when sea-eco is
importable, without the caller converting anything. This module makes that
possible in one place, following the ``seashells`` pattern established by
rayTEM (``src/pySEA/rayTEM/seashells.py``) and the third-party plugin guide in
``sea-eco/examples/example_3rd_party/``:

* sea-eco present — the real ``Signal``/``SignalSet``/``SignalCollection`` and
  friends are re-exported, and :func:`resolve` turns any PySlice object into
  the appropriate calibrated container.
* sea-eco absent — dummy stand-ins with the same names are exported, so
  PySlice imports and runs unchanged; :func:`resolve` raises one actionable
  error naming the install command, and nothing else in PySlice needs a guard.

Nothing else in PySlice imports ``pySEA`` directly. Add a type to the
resolution layer with :func:`register_resolver` rather than teaching another
module about sea-eco.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type

INSTALL_HINT = "Install sea-eco to get calibrated SEA objects: pip install 'pyslice[sea]'"

sea_available = False
_IMPORT_ERROR: Optional[str] = None

try:  # the one and only sea-eco import in PySlice
    from pySEA.sea_eco.architecture.base_structure import (  # noqa: F401
        Dimension,
        Dimensions,
        Metadata,
        SEAID,
        SEAFile,
        SEASerializable as _SEASerializable,
        Signal,
        SignalCollection,
        SignalQuantities,
        SignalSet,
        safe_decode,
    )
    from pySEA.sea_eco.signal_containers import (  # noqa: F401
        mark_atomic_structure,
        validate_atomic_structure,
    )

    sea_available = True
except Exception as exc:  # pragma: no cover - exercised only without sea-eco
    _IMPORT_ERROR = str(exc)

    class _SEASerializable:  # type: ignore[no-redef]
        """Stand-in for sea-eco's serialization base class.

        Keeps PySlice's class hierarchy intact when sea-eco is absent so
        result objects still construct; the SEA-specific methods warn instead
        of failing at import time.

        Methods
        -------
        to_sea(*args, **kwargs)
            Warn that sea-eco is required.
        from_sea(*args, **kwargs)
            Warn that sea-eco is required.
        """

        def to_sea(self, *args: Any, **kwargs: Any) -> None:
            """Warn that ``.sea`` export needs sea-eco.

            Returns
            -------
            None
            """
            warnings.warn(f"sea-eco is not installed, so .sea export is unavailable. {INSTALL_HINT}", stacklevel=2)

        def from_sea(self, *args: Any, **kwargs: Any) -> None:
            """Warn that ``.sea`` loading needs sea-eco.

            Returns
            -------
            None
            """
            warnings.warn(f"sea-eco is not installed, so .sea loading is unavailable. {INSTALL_HINT}", stacklevel=2)

    class Signal(_SEASerializable):  # type: ignore[no-redef]
        """Stand-in for sea-eco's ``Signal``, so PySlice classes still subclass.

        PySlice's result classes are declared as
        ``class WFData(PySliceSerial, Signal)``. Without a real class here the
        declaration itself would fail at import, so this dummy keeps the whole
        package importable with sea-eco absent; the SEA-specific methods warn
        (inherited from the dummy ``SEASerializable``).

        Attributes
        ----------
        name : str
            Display name, set by ``adopt_signal_state``.

        Methods
        -------
        to_sea(*args, **kwargs)
            Warn that sea-eco is required (inherited).
        """

    # Containers PySlice never subclasses, and helpers guarded by
    # ``if Dimensions is not None:`` at their call sites.
    SignalSet = SignalCollection = SEAFile = None  # type: ignore[assignment]
    Dimension = Dimensions = Metadata = SignalQuantities = SEAID = None  # type: ignore[assignment]
    safe_decode = None  # type: ignore[assignment]
    mark_atomic_structure = validate_atomic_structure = None  # type: ignore[assignment]

SEASerializable = _SEASerializable

_RESOLVERS: Dict[str, Callable[..., Any]] = {}
_BUILTINS_LOADED = False


def _ensure_builtin_resolvers() -> None:
    """Import PySlice's own resolvers once, so resolution needs no setup.

    Keeps :func:`resolve` implicit: a caller never has to import a resolver
    module to make ``resolve(trajectory)`` work, and this module still holds
    no PySlice-internal imports at module scope.

    Returns
    -------
    None
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED and "Trajectory" in _RESOLVERS:
        return
    try:
        from .atomic_structure import trajectory_to_atomic_structure

        register_resolver("Trajectory", trajectory_to_atomic_structure)
        _BUILTINS_LOADED = True
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"PySlice built-in SEA resolvers failed to load: {exc}", stacklevel=2)


def require_sea(feature: str = "this feature") -> None:
    """Raise an actionable error when sea-eco is required but missing.

    Parameters
    ----------
    feature : str, optional
        What the caller was trying to do, named in the message.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        Always, when sea-eco is unavailable; the message names the install
        command and the original import failure.

    See Also
    --------
    sea_available : Whether sea-eco was importable.
    """
    if sea_available:
        return
    detail = f" (import failed: {_IMPORT_ERROR})" if _IMPORT_ERROR else ""
    raise ImportError(f"{feature} requires sea-eco. {INSTALL_HINT}{detail}")


def register_resolver(type_name: str, resolver: Callable[..., Any]) -> None:
    """Register how one PySlice type resolves to a sea-eco container.

    Keyed by class name rather than by class object so registration never
    forces an import of the type being registered (which keeps this module
    free of PySlice-internal imports and avoids cycles).

    Parameters
    ----------
    type_name : str
        ``type(obj).__name__`` this resolver handles, e.g. ``"Trajectory"``.
    resolver : Callable
        Callable taking the object plus keyword arguments and returning a
        sea-eco ``Signal``, ``SignalSet``, or ``SignalCollection``.

    Returns
    -------
    None

    See Also
    --------
    resolve : Applies a registered resolver.

    Examples
    --------
    >>> register_resolver("MyResult", lambda obj, **kw: obj.as_signal())  # doctest: +SKIP
    """
    _RESOLVERS[type_name] = resolver


def resolve(obj: Any, **kwargs: Any) -> Any:
    """Return ``obj`` as a sea-eco container, resolving implicitly.

    Objects that already are sea-eco containers — which includes PySlice's
    ``WFData``, ``HAADFData``, and ``TACAWData``, since they subclass
    ``Signal`` — are returned unchanged. Registered types (``Trajectory``)
    are converted by their resolver. Bare arrays become minimal Signals so
    that every PySlice result has one uniform SEA form.

    Parameters
    ----------
    obj : Any
        PySlice object, sea-eco object, or array-like.
    **kwargs : Any
        Forwarded to the registered resolver (e.g. ``name``, ``build``).

    Returns
    -------
    Signal | SignalSet | SignalCollection
        The calibrated sea-eco form of ``obj``.

    Raises
    ------
    ImportError
        If sea-eco is not installed.
    TypeError
        If no resolver handles ``obj`` and it is not array-like.

    See Also
    --------
    register_resolver : Teach the layer a new type.

    Examples
    --------
    >>> resolve(trajectory)          # doctest: +SKIP
    <SignalCollection 'structure'>
    """
    require_sea("Resolving PySlice objects to SEA containers")

    if isinstance(obj, tuple(c for c in (Signal, SignalSet, SignalCollection) if c is not None)):
        return obj

    _ensure_builtin_resolvers()
    resolver = _RESOLVERS.get(type(obj).__name__)
    if resolver is not None:
        return resolver(obj, **kwargs)

    import numpy as np

    if isinstance(obj, np.ndarray) or hasattr(obj, "__array__"):
        array = np.asarray(obj)
        return Signal(data=array, name=kwargs.get("name", "array"))

    raise TypeError(
        f"No SEA resolver for {type(obj).__name__}. Register one with "
        "pyslice.data.seashell.register_resolver, or pass a Signal/SignalSet/"
        "SignalCollection, Trajectory, or array."
    )


def signal_defaults() -> Tuple[Tuple[str, Any], ...]:
    """Return the attribute defaults ``Signal.__init__`` would establish.

    PySlice's result classes build their arrays and calibration themselves and
    historically never called ``Signal.__init__``, which left sea-eco
    machinery (``Analysis``, ``Provenance``, dimension signatures) unset and
    made ordinary Signal methods fail. :func:`adopt_signal_state` applies
    these so a result object behaves as a first-class Signal.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        ``(attribute, default)`` pairs, evaluated lazily by the caller.

    See Also
    --------
    adopt_signal_state : Applies these defaults to an instance.
    """
    return (
        ("signal_type", None),
        ("dimensions_domain", "local"),
        ("is_lazy", False),
        ("_original_metadata", None),
        ("_dimensions_shared", None),
        ("_dimension_registry", None),
        ("_fold_state", None),
        ("detector", None),
        ("_parent_SignalSet", None),
    )


def adopt_signal_state(signal: Any, name: str) -> None:
    """Give a PySlice result object the Signal state it did not initialize.

    Sets a display ``name``, a fresh ``Provenance`` SEAID, an empty scalar
    ``SignalQuantities``, an ``AnalysisCollection``, and the dimension
    signature/role caches — i.e. what ``Signal.__init__`` would have set —
    without disturbing the arrays or ``Dimensions`` the class already built.
    A no-op when sea-eco is absent.

    Parameters
    ----------
    signal : Any
        A ``PySliceSerial``/``Signal`` instance under construction.
    name : str
        Display name for the result (e.g. ``"Wavefunction"``).

    Returns
    -------
    None

    See Also
    --------
    signal_defaults : The plain attribute defaults applied here.

    Notes
    -----
    Deliberately tolerant: any individual step that a future sea-eco version
    makes unnecessary is skipped rather than raised, because this runs inside
    every result constructor.
    """
    if not signal.__dict__.get("name"):
        signal.name = name
    if not sea_available:
        return

    from pySEA.sea_eco.architecture.base_structure import AnalysisCollection
    for attribute, default in signal_defaults():
        if not hasattr(signal, attribute):
            setattr(signal, attribute, default)
    if getattr(signal, "_signal_quantities", None) is None:
        signal._signal_quantities = SignalQuantities()
    try:
        if getattr(signal, "_Provenance", None) is None:
            signal.Provenance = SEAID()
    except Exception:  # pragma: no cover - identity is best-effort
        pass
    try:
        if getattr(signal, "_Analysis", None) is None and not hasattr(signal, "Analysis"):
            signal.Analysis = AnalysisCollection(name="Analysis")
    except Exception:  # pragma: no cover
        pass
    dimensions = getattr(signal, "_local_dimensions", None)
    if dimensions is not None:
        try:
            signal.dimension_signature = dimensions.get_names()
        except Exception:  # pragma: no cover
            pass
