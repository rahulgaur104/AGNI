# Tokamaks

A tokamak equilibrium is axisymmetric, so modes with different toroidal mode
numbers n do not couple. agnimhd therefore solves one n at a time on a
single toroidal plane, with the toroidal derivative replaced by i n (section
3.1 of the paper, after equation (28)). The matrix is smaller than that of a
three-dimensional solve by the number of toroidal nodes, and it is complex
Hermitian.

## Solving a DESC equilibrium

```python
import agnimhd

basis = agnimhd.Basis(48, 48, 1)        # radial, poloidal and one toroidal node
for n in (1, 2, 3, 4):
    config = agnimhd.AssemblyConfig(axisym=True, n_mode_axisym=n)
    gamma2, v, residual = agnimhd.solve("tokamak.h5", basis, assembly=config)
    print(n, float(gamma2), float(residual))
```

Each n has its own γ². The equilibrium is unstable when any of them is
positive, and the most unstable mode is the one with the largest. `v` is the
eigenvector and `residual` measures how well it is converged
([Checking a result](options.md#checking-a-result)). The command line has no
option for a single n, so tokamaks are solved from Python.

## The penalty

The default radial basis is Zernike, which is regular at the magnetic axis
([Choosing options](options.md#radial-basis)). The penalty gives the nodal
displacements off the Zernike polynomials a positive energy, so the most
unstable mode is made of Zernike polynomials. The default for a tokamak is 1.0.
Above that, γ² barely depends on the penalty. On the DSHAPE tokamak it
changes by 1.4 % from 1e2 to 1e4
([Benchmarks](benchmarks.md#zernike-jd-against-the-dense-solve)).

```python
basis = agnimhd.Basis(48, 48, 1, penalty=100.0)
```

How the penalty enters the energy is described in
[Theory](theory.md#penalty).

## The DSHAPE benchmark

The tokamak benchmark of the paper (section 5.2) is the DSHAPE equilibrium in
`tests/data`. Its settings are

```python
basis = agnimhd.Basis(96, 96, 1, mpol=4 * n, penalty=0.01)
config = agnimhd.AssemblyConfig(axisym=True, n_mode_axisym=n)
```

and agnimhd reproduces the paper's γ² for n = 2 to 5
([Benchmarks](benchmarks.md#dshape-tokamak)).

## From an EFIT g-file

An EFIT g-file (G-EQDSK) is first converted to a fixed-boundary DESC
equilibrium. The surface ψ<sub>N</sub> = `psi_n_edge` becomes the boundary, and the
pressure and ι = 1/q of the g-file are held fixed.

```python
eq = agnimhd.desc_from_eqdsk("g200245.02200", psi_n_edge=0.99)   # needs DESC
eq.save("g200245.02200.h5")
```

For the DIII-D g-file in `tests/data` this takes about 7 minutes on a CPU, so
save the equilibrium once and solve the `.h5` file as above. The poloidal flux
in the g-file must be in Wb/rad (COCOS 1 to 8). The conversion is a port of
[efit2desc](https://github.com/dpanici/efit2desc) by D. Panici, without its
OMFIT dependency.
