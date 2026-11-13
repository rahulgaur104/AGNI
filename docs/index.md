# agnimhd

agnimhd finds the most unstable ideal MHD mode of a tokamak or stellarator
equilibrium, and the derivative of its growth rate with respect to the
parameters of the equilibrium. It is the code of the AGNI paper (R. Gaur et
al., [arXiv:2608.01750](https://arxiv.org/abs/2608.01750), 2026), cited on
these pages as the paper. The package is under active development, and its
interface may change between versions.

## What it computes

A small displacement `xi` of the plasma changes the potential energy by
`delta W_p` (equation (16) of the paper) and carries the kinetic energy
`delta K` (equation (18)). The energy principle

```
delta W_p = -lambda delta K                                   (paper, equation (19))
```

is an eigenvalue problem for `xi` and `lambda`. agnimhd discretizes it on a
grid in PEST coordinates `(rho, theta, zeta)` and finds the largest `lambda`,
which is the squared growth rate `gamma^2` in units of `(v_A / a_N)^2`, with
`v_A` the Alfvén speed and `a_N` the minor radius. The code returns it as
`gamma2`.

**A positive `gamma2` means the equilibrium is unstable.** A negative one means
the mode oscillates and the equilibrium is stable to it.

## A first run

agnimhd reads DESC equilibria directly.

```bash
pip install "desc-opt==0.17.3" agnimhd
agnimhd solve my_equilibrium.h5 --res 24,12,8
```

The second line evaluates the equilibrium on 24 radial, 12 poloidal and 8
toroidal nodes, prints `gamma2` for each family of toroidal modes, and ends
with the verdict `UNSTABLE` or `stable`.

## Next steps

- [Installation](installation.md) covers GPUs and installing without DESC.
- The [Tutorial](notebooks/stability_of_an_equilibrium.ipynb) solves a
  stellarator step by step in about four minutes on one CPU core, without DESC.
- [Stellarators](stellarators.md) and [Tokamaks](tokamaks.md) show the solve
  for each kind of device.
- [Choosing options](options.md) explains the resolution, the eigensolver and
  the shift first, then the remaining settings.
- [DESC coupling](desc.md) uses `gamma2` and its derivative as an objective in a
  DESC optimization.
- [Interface](interface.md) lists the arrays agnimhd needs from an equilibrium
  code other than DESC.
- [Theory](theory.md) follows the discretization and the eigensolvers equation
  by equation.

## Citing

R. Gaur, S. Patil, P. Gupta, D. Patch, T. Qian, *AGNI: A differentiable MHD
stability solver & optimizer for magnetic confinement fusion devices*,
[arXiv:2608.01750](https://arxiv.org/abs/2608.01750) (2026). The PDF of v3 is in
the repository at `docs/paper/agni_arxiv_2608.01750v3.pdf`. BibTeX is in the
README and `CITATION.cff`.
