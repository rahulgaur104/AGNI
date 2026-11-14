"""Free-boundary vacuum pipeline.

Private to :mod:`agnimhd.adapters.desc_vacuum`, which is its only user. Imports DESC
at module level, like every module under ``adapters``; ``import agnimhd`` never loads
it. Each module says where it came from.
"""
