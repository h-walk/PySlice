PySlice Documentation
=====================

PySlice (import ``pyslice``) is the pySEA ecosystem's GPU-accelerated multislice
simulation engine: TEM/STEM/4D-STEM imaging and diffraction, plus vibrational
EELS via the TACAW method computed from molecular-dynamics trajectories with
universal machine-learned potentials.

When sea-eco is importable, **every PySlice result is a sea-eco container** —
no conversion step. See :doc:`guides/sea_results` for the user-facing view and
:doc:`sea-weeds/resolution_layer` for the design.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/getting_started
   guides/sea_results
   guides/structures
   guides/prompted_simulations
   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks

   guides/examples

.. toctree::
   :maxdepth: 2
   :caption: AI Tools

   ai_tools/install
   ai_tools/mcp
   ai_tools/skills
   ai_tools/subagents

.. toctree::
   :maxdepth: 2
   :caption: Into the SEA-weeds (For developers)

   sea-weeds/index
   sea-weeds/resolution_layer
   sea-weeds/signal_containers
   sea-weeds/structure_building
   sea-weeds/agentic_surface
   conformance/signal-containers

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
