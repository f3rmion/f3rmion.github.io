## Efficiently calculate quantum-chemical features

In this blog post, I'd like to describe how you can make use of `kallisto` as a Python package for the efficient calculation of quantum-chemical features.
Theoretical bits about kallisto are documented in its [GitBook](https://ehjc.gitbook.io/kallisto/) where we mostly describe its application as a command-line interface (CLI).

Today, we want to go beyond the CLI-usage and concentrate us on using it as a Python package to retrieve some very useful features.
Follow the steps below to get a feeling for how to use this package.
Note that we assume a freshly created virtual environment, e.g., _via_ Conda

```markdown
$ conda create --name kallisto python=3.10
```

Activate that virtual environment and continue.

```markdown
$ conda activate kallisto
```

## Setup `kallisto`

We install the package via `pip`, which gets the pre-built package from [PyPI](https://pypi.org/project/kallisto/).

```markdown
$ pip install kallisto
...
Installing collected packages: numpy, click, scipy, kallisto
Successfully installed click-8.1.7 kallisto-1.0.10 numpy-1.26.4 scipy-1.12.0
```

`kallisto` depends on some scientific Python libraries like _numpy_, _scipy_, and uses _click_ for its CLI.

Next, we verify its installation by requesting some help.
This will print out some options and arguments for the CLI.

```markdown
$ kallisto --help
Usage: kallisto [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

kallisto calculates quantum mechanically derived atomic features.

Please check out the documentation (https://ehjc.gitbook.io/kallisto/).

Always cite:

E. Caldeweyher J. Open Source Softw., 2021, 6, 3050.
(https://doi.org/10.21105/joss.03050)

Options:
--silent
--shift INTEGER
--help Show this message and exit.

Commands:
alp Static atomic polarizabilities in Bohr^3.
bonds Get information about covalent bonding partner.
cns Atomic coordination numbers.
eeq Electronegativity equilibration atomic partial charges.
exs Exchange a substrate within a transition metal complex with...
lig Get all substructures (or ligands) that are bound to the center...
prox Atomic proximity shells.
rms Calculate the root mean squared deviation between two structures...
sort Sort input geoemtry according to connectivity.
stm Calculate sterimol descriptors using kallisto van der Waals radii.
vdw Charge-dependent atomic van der Waals radii in Bohr.
```

However, today we want to test it as a package!
Hence, let's try to Python import this package

```python
# this should work without errors
import kallisto
```

If this works without errors, we are ready to calculate some features!

## Build a `kallisto` Molecule

Let's start simple by creating an Alanine-Glycine molecule in this `xmol` format.

```markdown
> cat alanine-glycine.xyz
> 20
> Alanine-Glycine
> C 2.081440 0.615100 -0.508430
> C 2.742230 1.824030 -1.200820
> N 4.117790 1.799870 -1.190410
> C 4.943570 2.827040 -1.822060
> C 6.440080 2.569360 -1.637600
> O 7.351600 3.252270 -2.069090
> N 0.610100 0.695090 -0.538780
> O 2.095560 2.724940 -1.739670
> O 6.705220 1.463410 -0.897460
> H 0.303080 1.426060 0.103770
> H 0.338420 1.050680 -1.460480
> C 2.488753 -0.593400 -1.198448
> H 2.416500 0.557400 0.532050
> H 4.614100 1.081980 -0.670550
> H 4.699850 3.794460 -1.373720
> H 4.722890 2.844690 -2.894180
> H 7.687400 1.448620 -0.860340
> H 2.029201 -1.457008 -0.719999
> H 2.170233 -0.542411 -2.238576
> H 3.572730 -0.688405 -1.154998
```

Next, we create a `kallisto` molecule, which gives us a rich ensemble of features through methods.

```python
import kallisto.reader.strucreader as ksr

# define the file name
fname = "alanine-glycine.xyz"

# construct a kallisto molecule
kmol = ksr.constructMolecule(geometry=fname, out=None)

# calculate coordination numbers
cn_type = "cov"
cns = kmol.get_cns(cn_type)

# calculate proximity shells
# where size defines the
# inner and outer border of the shells
size = (3, 2)
proxs = kmol.get_prox(size)

# atomic partial charges
# elecotronegativity equilibration
total_charge = 0
eeqs = kmol.get_eeq(total_charge)

# atomic charge-dependent static polarizabilities
# TD-PBE38/d-aug-def2-QZVP references
alps = kmol.get_alp(total_charge)

# atomic charge-dependent van-der-Waals radii in Bohr
# we parametrized this to match different references:
# - Rahm: DOI:10.1002/chem.201602949
# - Truhlar: DOI:10.1021/jp8111556
vdw_type = "rahm"
vdws = kmol.get_vdw(total_charge, vdw_type, 1.0)
```

Above we calculate the following features:

- [Covalent coordination numbers](https://ehjc.gitbook.io/kallisto/features/cns)
- [Proximity shells](https://ehjc.gitbook.io/kallisto/features/prox)
- [Atomic partial charges](https://ehjc.gitbook.io/kallisto/features/eeq)
- [Atomic static polarizabilities](https://ehjc.gitbook.io/kallisto/features/alp)
- [Atomic van-der-Waals radii](https://ehjc.gitbook.io/kallisto/features/vdw)

I hope this helps you while using `kallisto` as a package.

Cheers!
ehjc
