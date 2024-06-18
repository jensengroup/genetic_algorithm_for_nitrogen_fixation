# Discovery of nitrogen fixation catalysts with genetic algorithms

Code repository for the paper [Discovery of molybdenum based nitrogen fixation catalysts with genetic algorithms](https://pubs.rsc.org/en/Content/ArticleLanding/2024/SC/D4SC02227K).

1. [Run](#how-to-run)
1. [Parameters](#parameters)

## How to run

To setup an environment for running the GA, activate a conda environment with pip installed and run:

```bash
pip install .
```

Then ensure you have installed [`xtb`](https://xtb-docs.readthedocs.io/en/latest/setup.html) in the environment.

To do a simple GA run, activate the environment and run:

```
python genetic_algorithm_driver.py --scoring_func calculate_score --partition kemi1 --generations 1 --filename data/ligands.smi --debug --population_size 4 --molecules_per_chunk 2 --cpus_per_chunk 2 --sa_screening --CovalentLigands 3 --DativeLigands 1 --opt loose --n_confs 2
```

## Input driver arguments

A list of possible arguments and their discriptions are seen below:

| Arg                       | Description                                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| `-h` or `--help`          | Prints help message.                                                                                                 |
| `--population_size`       | Sets the size of the population pool.                                                                                |
| `--n_confs`               | Sets how many conformers to generate for each molecule.                                                              |
| `--num_rotatable_bonds`   | Set the limit for the number of rotatable bonds for new children.                                                    |
| `--max_size`              | Maximum number of heavy atoms allowed in crossover                                                                   |
| `--min_size`              | Minimum number of heavy atoms allowed in crossover                                                                   |
| `--DativeLigands`         | Number of dative ligands in each catalyst                                                                            |
| `--CovalentLigands`       | Number of covalent ligands in each catalyst                                                                          |
| `--BidentateLigands`      | Number of bidentate ligands in each catalyst                                                                         |
| `--molecules_per_chunk`   | Parallelization option. How many molecules to put in each SLURM job                                                  |
| `--cpus_per_chunk`        | Parallelization option. How many cpus to reserve for each SLURM job                                                  |
| `--memory_per_chunk`      | Parallelization option. How much memory per cpu in a SLURM job                                                       |
| `--partition`             | Cluster to run on                                                                                                    |
| `--debug`                 | If set the starting population is a set of 4 small molecules that can run fast locally. Used for debugging.          |
| `--find_connection_point` | Pre-screen ligand candidates for connection points                                                                   |
| `--load_from_pickle`      | Load starting population from pickle file                                                                            |
| `--minimize_score`        | Minimize the score obtained from scoring function                                                                    |
| `--generations`           | How many evolution cycles of the population is performed.                                                            |
| `--n_tries`               | Sets how many times the GA is restarted. Can be used to run multiple GA runs in a single submission.                 |
| `--scoring_function`      | Chose one of the specified scoring functions to run                                                                  |
| `--sa_screening`          | Decides if synthetic accessibility score is enabled. Highly recommended to turn this on.                             |
| `--filename`              | Path to the database extract to create starting population.                                                          |
| `--output_dir`            | Sets output directory for all files generated during generations.                                                    |
| `--scratch`               | Path to the database extract to create starting population.                                                          |
| `--cleanup`               | If enabled, all scoring files are removed after scoring. Only the optimized structures and their energies are saved. |
| `--method`                | Which gfn method to use.                                                                                             |
| `--timeout_min`           | How many minutes each slurm job is allowed to run                                                                    |
| `--rms_prune`             | Pruning value for RDKit conformer embedding                                                                          |
| `--energy_cutoff`         | Sets energy cutoff on the conformer filtering.                                                                       |
| `--opt`                   | Set optimization convergence criteria for xTB.                                                                       |

# Authors

**Magnus Strandgaard**<sup>1</sup>

<sup>1</sup> Department of Chemistry, University of Copenhagen, 2100 Copenhagen Ø, Denmark.
Email: _mastr@chem.ku.dk_.
