"""Driver script for running a GA algorithm.

Written by Magnus Strandgaard 2023

Example:
    How to run:

        $ python genetic_algorithm_driver.py --args
"""
import argparse
import logging
import os
import pathlib
import sys
import time
from pathlib import Path, PosixPath
from typing import NoReturn

from catalystGA.utils import MoleculeOptions

from classes.GeneticAlgorithm import GeneticAlgorithm
from classes.schrock_paralell import Schrock
from utils.utils import get_git_revision_short_hash

# Initialize logger
_logger: logging.Logger = logging.getLogger(__name__)


def get_arguments(arg_list=None) -> argparse.Namespace:
    """

    Args:
        arg_list: Automatically obtained from the commandline if provided.
        Otherwise default arguments are used

    Returns:
        parser.parse_args(arg_list)(Namespace): Dictionary like class that contain the driver arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run GA algorithm", fromfile_prefix_chars="+"
    )

    ### Population
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Sets the size of population pool.",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=2,
        help="How many conformers to generate",
    )
    parser.add_argument(
        "--num_rotatable_bonds",
        type=int,
        default=20,
        help="Setting the limit for num rotatable bonds for each FULL catalyt",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=100,
        help="Sets the max number of heavy atoms in the FULL catalyst",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=2,
        help="Sets the min number of heavy atoms in the FULL catalyst",
    )
    ### Ligand distribution
    parser.add_argument(
        "--DativeLigands",
        type=int,
        default=0,
        help="How many Dative ligands to use",
    )
    parser.add_argument(
        "--CovalentLigands",
        type=int,
        required=True,
        help="""How many X-ligands to use""",
    )
    parser.add_argument(
        "--BidentateLigands",
        type=int,
        default=0,
        help="How many Bidentate ligands to use",
    )
    ### Computational
    parser.add_argument(
        "--molecules_per_chunk",
        type=int,
        default=1,
        help="How many molecules to put in each chunk",
    )
    parser.add_argument(
        "--cpus_per_chunk",
        type=int,
        default=1,
        help="Number of cores for each chunk of molecules",
    )
    parser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="500MB",
        help="Mem per cpu allocated for submitit job",
    )
    parser.add_argument(
        "--partition",
        choices=[
            "kemi1",
            "xeon16",
            "xeon24",
            "xeon40",
        ],
        required=True,
        help="""Choose partition to run on""",
    )
    ### Diverse GA args
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--find_connection_point",
        action="store_true",
        help="Pre-screeen ligand candidate for connection points.",
    )
    parser.add_argument(
        "--load_from_pickle",
        default=False,
        type=pathlib.Path,
        help="Load starting population from pickle",
    )
    parser.add_argument(
        "--minimize_score",
        action="store_false",
        help="Minimize the score in the scoring function.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="How many times is the population optimized",
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        default=1,
        help="How many overall runs of the GA",
    )
    parser.add_argument(
        "--scoring_function",
        choices=["calculate_score_logP", "calculate_score"],
        required=True,
        help="Choose one of the specified scoring functions to be run.",
    )
    parser.add_argument(
        "--sa_screening",
        dest="sa_screening",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="./data/ZINC_250k.smi",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=PosixPath,
        default="debug/",
        help="Directory to put output files",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default="SCRATCH",
        help="Directory to put scratch files",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean xtb files after calulation"
    ),

    ### XTB related options
    parser.add_argument(
        "--method",
        type=str,
        default="2",
        help="gfn method to use",
    )
    parser.add_argument(
        "--timeout_min",
        type=int,
        default=12,
        help="Minutes before timeout for each slurm job.(Each chunk of molecules)",
    )
    parser.add_argument(
        "--rms_prune",
        type=float,
        default=0.25,
        help="Embedding pruning",
    )
    parser.add_argument(
        "--energy_cutoff",
        type=float,
        default=0.0319,
        help="Cutoff for conformer energies",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="tight",
        help="Opt convergence criteria for XTB",
    )
    return parser.parse_args(arg_list)


def main() -> NoReturn:
    """Main function that starts the GA."""
    args = get_arguments()

    # Get arguments as dict and add scoring function to dict.
    args_dict = vars(args)

    # Create list of dicts for the distributed GAs
    root = args.output_dir

    # Set Options for Molecule
    mol_options = MoleculeOptions(
        individual_type=Schrock,
        max_size=args.max_size,
        min_size=args.min_size,
        num_rotatable_bonds=args.num_rotatable_bonds,
    )

    # Run the GA
    for i in range(args.n_tries):
        # Start the time
        t0 = time.time()
        # Create output_dir
        args_dict["output_dir"] = root / f"gen_{i}_{time.strftime('%Y-%m-%d_%H-%M')}"
        Path(args_dict["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(args_dict["output_dir"], "printlog.txt"), mode="w"
                ),
                logging.StreamHandler(),  # For debugging. Can be removed on remote
            ],
        )

        # Log current git commit hash
        logging.info("Current git hash: %s", get_git_revision_short_hash())

        # Log the argparse set values
        logging.info("Input args: %r", args)

        # Set how many cpus per mol
        args_dict["cpus_per_mol"] = (
            args_dict["cpus_per_chunk"] // args_dict["molecules_per_chunk"]
        )

        # Initialize GA
        ga = GeneticAlgorithm(args=args_dict, mol_options=mol_options)

        # Run the GA
        ga.run()

        # Final output handling and logging
        t1 = time.time()
        logging.info(f"# Total duration: {(t1 - t0) / 60.0:.2f} minutes")

        logging.info(
            """ GA RUN IS DONE


         /$$$$$$$  /$$                     /$$                 /$$$$$$$                                /$$
        | $$__  $$|__/                    | $$                | $$__  $$                              | $$
        | $$  \ $$ /$$  /$$$$$$$  /$$$$$$$| $$$$$$$           | $$  \ $$  /$$$$$$   /$$$$$$$  /$$$$$$$| $$$$$$$
        | $$$$$$$ | $$ /$$_____/ /$$_____/| $$__  $$          | $$$$$$$  |____  $$ /$$_____/ /$$_____/| $$__  $$
        | $$__  $$| $$|  $$$$$$ | $$      | $$  \ $$          | $$__  $$  /$$$$$$$|  $$$$$$ | $$      | $$  \ $$
        | $$  \ $$| $$ \____  $$| $$      | $$  | $$          | $$  \ $$ /$$__  $$ \____  $$| $$      | $$  | $$
        | $$$$$$$/| $$ /$$$$$$$/|  $$$$$$$| $$  | $$ /$$      | $$$$$$$/|  $$$$$$$ /$$$$$$$/|  $$$$$$$| $$  | $$
        |_______/ |__/|_______/  \_______/|__/  |__/| $/      |_______/  \_______/|_______/  \_______/|__/  |__/
                                                    |_/


         /$$   /$$           /$$                                 /$$$$$$$                                /$$
        | $$  | $$          | $$                                | $$__  $$                              | $$
        | $$  | $$  /$$$$$$ | $$$$$$$   /$$$$$$   /$$$$$$       | $$  \ $$  /$$$$$$   /$$$$$$$  /$$$$$$$| $$$$$$$
        | $$$$$$$$ |____  $$| $$__  $$ /$$__  $$ /$$__  $$      | $$$$$$$  /$$__  $$ /$$_____/ /$$_____/| $$__  $$
        | $$__  $$  /$$$$$$$| $$  \ $$| $$$$$$$$| $$  \__/      | $$__  $$| $$  \ $$|  $$$$$$ | $$      | $$  \ $$
        | $$  | $$ /$$__  $$| $$  | $$| $$_____/| $$            | $$  \ $$| $$  | $$ \____  $$| $$      | $$  | $$
        | $$  | $$|  $$$$$$$| $$$$$$$/|  $$$$$$$| $$            | $$$$$$$/|  $$$$$$/ /$$$$$$$/|  $$$$$$$| $$  | $$
        |__/  |__/ \_______/|_______/  \_______/|__/            |_______/  \______/ |_______/  \_______/|__/  |__/


                                                                                                                  """
        )

    # Ensure the program exists when running on the frontend.
    sys.exit(0)


if __name__ == "__main__":
    main()
