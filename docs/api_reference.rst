API Reference
=============

Generated from NumPy-style docstrings in ``src/pyslice``. Regenerate with the
command recorded in :doc:`sea-weeds/index` ("Building the docs"); heavy
optional dependencies (OVITO, torch, ORB, the MCP SDK) are mocked by
``docs/conf.py`` so this section builds in any environment.

The data layer
--------------

.. automodule:: pyslice.data.seashell
   :members:

.. automodule:: pyslice.data.atomic_structure
   :members:

Structure input and building
----------------------------

.. automodule:: pyslice.io.build
   :members:

.. automodule:: pyslice.io.databases
   :members:

Trajectories
------------

.. automodule:: pyslice.multislice.trajectory
   :members:

Agent surface
-------------

.. autoclass:: pyslice.mcp.service.PySliceService
   :no-members:

Each ``pyslice_*`` MCP tool is one method on this class; the class docstring
above lists them. Per-tool parameters are documented by the server itself —
call ``pyslice_get_conventions``, and see :doc:`ai_tools/mcp`. The tool inputs
are pydantic models in the same module and are not duplicated here.
