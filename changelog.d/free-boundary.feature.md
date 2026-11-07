Free boundary. ``EquilibriumData.phi_matrix`` (the boundary's vacuum response,
``FORMAT_VERSION`` 4) makes every solver free-boundary: ``xi^rho`` is kept on the
outer shell and the vacuum energy is added, through a kernel projected onto its
positive semidefinite part so it is never negative. ``Basis(...,
free_boundary=True)`` rescales any radial basis so its outer node is ``rho = 1``,
and ``from_desc`` / ``load(...).evaluate`` then compute DESC's
``phi_matrix_pest`` there (``agnimhd.adapters.desc_vacuum``, with the
``phi_*`` options of DESC's ``FinitenStability``); ``agnimhd solve
--free-boundary``. ``assemble.vacuum_energy`` reports the vacuum energy of a
mode.
