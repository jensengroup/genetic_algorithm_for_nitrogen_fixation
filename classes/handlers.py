" Module containing utility classes for handling input and output of the GA"
import logging
import os
import pickle
import random
import shutil
import time
import uuid
from pathlib import PosixPath
from typing import Dict, List, Optional, Union

import pandas as pd
import submitit
from catalystGA import BidentateLigand, CovalentLigand, DativeLigand, Metal
from catalystGA.utils import MoleculeOptions

# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit import Chem
from submitit.core.utils import FailedJobError
from tabulate import tabulate

from classes.schrock_paralell import Schrock
from utils.utils import chunks, read_file_tolist, renamed_load

_logger = logging.getLogger(__name__)


class OutputHandler:
    """Class responsible for printing to output files and handle saving of
    molecules."""

    def __init__(
        self, args: Dict[str, Union[int, float, str, bool, PosixPath]]
    ) -> None:
        self.args = args

    @staticmethod
    def save(
        molecules: List[Schrock],
        directory: Optional[PosixPath] = None,
        name: Optional[str] = None,
    ) -> None:
        """Save instance to file for later retrieval."""
        filename = os.path.join(directory, name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "wb+") as output:
            pickle.dump(molecules, output, pickle.HIGHEST_PROTOCOL)

    def write_out(
        self,
        molecules: List[Schrock],
        name: Optional[str] = None,
        genid: Optional[int] = None,
    ) -> None:
        self.molecules = molecules
        with open(self.args["output_dir"] / name, "a") as f:
            f.write(f"\n Generation {genid} \n")
            f.write(self.print(pass_text=True) + "\n")

    def print(self, pass_text: Optional[bool] = None) -> str:
        """Print nice table of population attributes."""
        table = []
        relevant_props = [
            "idx",
            "fitness",
            "score",
            "energy",
            "sa_score",
            "smiles",
        ]
        for individual in self.molecules:
            props = individual.get_props()
            try:
                props["smiles"] = individual.smiles
                props["sa_score"] = individual.sa_score
            except Exception:
                props["smiles"] = "Not available"
                props["sa_score"] = "Not available"
                _logger.warning(
                    f"Smiles or SA score not retrieved for {individual.idx}. Check assmebly for errors"
                )

            property_row = [props[key] for key in relevant_props]
            table.append(property_row)
        txt = tabulate(table, headers=relevant_props)
        if pass_text:
            return txt

    def pop2pd(
        self,
        columns=["cut_idx", "score", "energy", "sa_score", "smiles"],
    ):
        """Get dataframe of population."""
        df = pd.DataFrame(
            list(map(list, zip(*[self.get(prop) for prop in columns]))),
            index=pd.MultiIndex.from_tuples(
                self.get("idx"), names=("generation", "individual")
            ),
        )
        df.columns = columns
        return df

    def pop2pd_dft(self):
        columns = [
            "smiles",
            "idx",
            "cut_idx",
            "score",
            "energy",
            "dft_singlepoint_conf",
            "min_confs",
        ]
        """Get dataframe of population."""
        df = pd.DataFrame(list(map(list, zip(*[self.get(prop) for prop in columns]))))
        df.columns = columns
        return df


class DataLoader:
    """Class handling the loading of the starting population."""

    def __init__(
        self,
        args: Dict[str, Union[int, float, str, bool, PosixPath]],
        mol_options: MoleculeOptions,
    ) -> None:
        self.args = args
        self.mol_options = mol_options
        self.filename = args["filename"]

    def load_data(self) -> List[Schrock]:
        "Wrapper function to call given load functions"
        if self.args["load_from_pickle"]:
            population = self.load_from_pickle()
        elif self.args["debug"]:
            population = self.load_debug()
        else:
            population = self.load_from_file()

        return population

    def load_from_file(self) -> List[Schrock]:
        """Create starting population from csv file."""
        metals_list = [Metal("Mo")]
        metal = random.choice(metals_list)
        # Get mol generator from file
        mols = read_file_tolist(self.filename)
        population = []

        dative_ligand = None
        covalent_ligand = None
        while True:
            mol = random.choice(mols)
            if mol.GetNumHeavyAtoms() < self.args["max_size"]:
                try:
                    # If we want bidentate ligand select one at random first.
                    if self.args["BidentateLigands"] > 0:
                        suppl = Chem.SDMolSupplier(
                            "data/bidentate_uncharged.sdf", removeHs=False
                        )
                        bidentate_mol = random.choice(suppl)
                        bidentate_ligand = BidentateLigand(bidentate_mol)
                    if not dative_ligand:
                        dative_ligand = DativeLigand(mol, smarts_match=True)
                        continue
                    else:
                        if not mol.GetSubstructMatch(Chem.MolFromSmarts("[#6&v2H0]")):
                            covalent_ligand = CovalentLigand(mol)
                        else:
                            continue

                    if not (dative_ligand and covalent_ligand):
                        continue

                    constructor = (
                        [covalent_ligand for _ in range(self.args["CovalentLigands"])]
                        + [
                            bidentate_ligand
                            for _ in range(self.args["BidentateLigands"])
                        ]
                        + [dative_ligand for _ in range(self.args["DativeLigands"])]
                    )

                    # Try assemble to ensure not wrong connection ids that give kekulization error gets passed through. # TODO Better fix
                    self.mol_options.individual_type(metal, constructor).assemble()
                    population.append(
                        self.mol_options.individual_type(metal, constructor)
                    )

                    # Reset ligands if a valid catalyst was found.
                    covalent_ligand = None
                    dative_ligand = None

                except Exception as e:
                    _logger.info(
                        f"Error in creating constructor. Skipping {Chem.MolToSmiles(mol)}"
                    )
                    _logger.warning(f"Traceback for {Chem.MolToSmiles(mol)}: {e}")
                    continue
            if len(population) == self.args["population_size"]:
                break
        # Enumerate the catalysts in the population.
        for i, ind in enumerate(population):
            ind.idx = (0, i)
        return population

    def load_debug(self) -> List[Schrock]:
        "Helper function for loading simple starting populations for debugging."

        population = []

        # Smiles with primary amines and corresponding cut idx
        mols = ["CCN", "CCCC", "CCCCO"]
        mols = [Chem.MolFromSmiles(m) for m in mols]

        metals_list = [Metal("Mo")]
        metal = random.choice(metals_list)

        dative_ligand = None
        covalent_ligand = None
        while True:
            mol = random.choice(mols)
            try:
                # If we want bidentate ligand select one at random first.
                if self.args["BidentateLigands"] > 0:
                    suppl = Chem.SDMolSupplier(
                        "data/bidentate_uncharged.sdf", removeHs=False
                    )
                    bidentate_mol = random.choice(suppl)
                    bidentate_ligand = BidentateLigand(bidentate_mol)
                if not dative_ligand:
                    dative_ligand = DativeLigand(mol, smarts_match=True)
                    if not dative_ligand.connection_atom_id:
                        dative_ligand = None
                    continue
                else:
                    # Avoid assignign carbene ligand as CovalentLigand.
                    if not mol.GetSubstructMatch(Chem.MolFromSmarts("[#6&v2H0]")):
                        covalent_ligand = CovalentLigand(mol)
                        if not covalent_ligand.connection_atom_id:
                            covalent_ligand = None
                    else:
                        continue

                if not (dative_ligand and covalent_ligand):
                    continue

                constructor = (
                    [covalent_ligand for _ in range(self.args["CovalentLigands"])]
                    + [bidentate_ligand for _ in range(self.args["BidentateLigands"])]
                    + [dative_ligand for _ in range(self.args["DativeLigands"])]
                )

                # Try assmble to ensure not wrong connection ids that giv kekulization error gets passed through. # TODO Better fix
                self.mol_options.individual_type(metal, constructor).assemble()
                population.append(self.mol_options.individual_type(metal, constructor))
                covalent_ligand = None
                dative_ligand = None

            except Exception as e:
                _logger.info(
                    f"Error in creating constructor. Skipping {Chem.MolToSmiles(mol)}"
                )
                _logger.warning(f"Traceback for {Chem.MolToSmiles(mol)}: {e}")
                continue
            if len(population) == self.args["population_size"]:
                break
        for i, ind in enumerate(population):
            ind.idx = (0, i)
        return population

    ### CURENTLY NOT USED FOR ANYTHING
    def load_from_pickle(self):
        """Create starting population from already created list of Schrock
        objects."""
        with open(self.args["load_from_pickle"], "rb") as f:
            population = renamed_load(f)
        # Clean population and ensure
        # The previous information is still stored.
        for i, catalyst in enumerate(population):
            catalyst.clean()
            catalyst.old_idx = catalyst.idx
            catalyst.idx = (0, i)

        return population


class SubmititHandler:
    """Handling of SLURM submissions and output."""

    def __init__(self, args):
        self.args = args

    def wrap_scoring_chunks(self, chunk, args):
        """Helper function to control multiprocessing of chunks of
        molecules."""
        _logger.info(
            f"Scoring chunk with individuals {[individual.idx for individual in chunk]}"
        )

        """If there is only one molecule in each chunk, then the jobs only has one calculation to run"""
        if len(chunk) == 1:
            ind = self.wrap_scoring(individual=chunk[0], args=args)
            processed_chunk = [ind]
        else:
            # Submit each molecules as its own process. The error handling is done in the child process!
            with Pool(
                processes=args["molecules_per_chunk"]
            ) as pool:  # start worker processes
                processed_chunk = pool.map(
                    self.wrap_scoring, [(mol, args) for mol in chunk]
                )

        return processed_chunk

    def find_donor_atom(self, individual, reference_smiles="[Mo]", args=None):
        """Find the donor atom of all unique ligands in a catalyst."""
        # Setup logging to stdout
        _logger = logging.getLogger("ligand")
        _logger.setLevel(logging.INFO)
        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        )
        _logger.addHandler(stream)

        # Setup scrach directory
        scratch = (
            args["output_dir"] / args["scratch"] / f"find_donor_{str(uuid.uuid4())}"
        )
        scratch.mkdir(parents=True, exist_ok=True)

        # We want the same connection point on all similar ligands.
        unique = set(individual.ligands)

        for i, ligand in enumerate(unique):
            # Construct args dict
            1 if type(ligand) == DativeLigand else 0
            calc_args = {"charge": args["charge"], "uhf": 0}
            calc_dir = (
                scratch
                / f"{individual.idx[0]:03d}_{individual.idx[1]:03d}_{i}_{ligand.__class__.__name__}"
            )
            calc_dir.mkdir(exist_ok=True)

            ligand.find_donor_atom(
                smarts_match=False,
                reference_smiles=reference_smiles,
                n_cores=args["cpus_per_mol"],
                xtb_args=calc_args,
                calc_dir=calc_dir,
                numConfs=args["n_confs"],
            )
            for elem in individual.ligands:
                if type(elem) == type(ligand):
                    elem.connection_atom_id = ligand.connection_atom_id
        if args["cleanup"]:
            shutil.rmtree(scratch.resolve())
        return

    def wrap_scoring(self, inp):
        "Helper function that cals the scoring function on the object."
        individual, args = inp
        if args["find_connection_point"]:
            _logger.info(f"Finding connection point for {individual.idx}")
            self.find_donor_atom(
                individual,
                reference_smiles=Chem.MolToSmiles(individual.metal.atom),
                args=args,
            )
            _logger.info("Done with connection check")

        _logger.info(f"Scoring individual {individual.idx}")
        individual.dispatcher[args["scoring_function"]](args)
        return individual

    def calculate_scores(self, population: List[Schrock], gen_id: int) -> List[Schrock]:
        """Calculates scores for all individuals in the population by
        submittting jobs to cluster.

        Args:
            population (List): List of individuals

        Returns:
            population: List of individuals with scores
        """
        scoring_temp_dir = (
            self.args["output_dir"]
            / f"scoring_{time.strftime('%Y%m%d-%H%M%S')}_{gen_id}"
        )
        executor = submitit.AutoExecutor(folder=scoring_temp_dir)
        executor.update_parameters(
            name=f"sc_{gen_id}",
            cpus_per_task=self.args["cpus_per_chunk"],
            slurm_mem_per_cpu="500MB",
            timeout_min=self.args["timeout_min"]
            + 2,  # This is a timeout buffer. Submitit cancels 2 minutes before what is asked for sometimes
            slurm_partition=self.args["partition"],
            slurm_array_parallelism=20,
        )

        # Get list of lists that make up the scoring chunks.
        chunks_of_molecules = list(chunks(population, self.args["molecules_per_chunk"]))

        jobs = executor.map_array(
            self.wrap_scoring_chunks,
            chunks_of_molecules,
            [self.args] * len(chunks_of_molecules),
        )
        # read results, if job terminated with error then return individual without score
        new_population = []
        dir = scoring_temp_dir / "chunks"
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Handle jobs output.
        for i, job in enumerate(jobs):
            error = "Normal termination"
            # Catch any exceptions raised in the scoring function.
            try:
                chunk_result = job.result()
            except FailedJobError as e:
                error = f"Exception for chunk id {i}\nTraceback: {e}\n"
                error += f"{job.stderr()}"
                chunk_result = chunks_of_molecules[i]
                # Save the individual chunks to prevent loss of data if script somehow fails
                with open(dir / f"chunk_{i}.pkl", "wb+") as output:
                    pickle.dump(chunk_result, output, pickle.HIGHEST_PROTOCOL)
            finally:
                # Set the error string returned for a catalyst for debugging.
                for mol in chunk_result:
                    mol.error = error
            population = new_population
            new_population.extend(chunk_result)

        return population
