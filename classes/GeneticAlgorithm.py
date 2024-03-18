"""Written by Magnus Strandgaard 2023."""

import copy
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Union

import catalystGA.components
from catalystGA import BidentateLigand, CovalentLigand, DativeLigand
from catalystGA.ga import GA
from catalystGA.reproduction_utils import graph_crossover, graph_mutate
from catalystGA.utils import MoleculeOptions
from rdkit import Chem

from classes.handlers import DataLoader, OutputHandler, SubmititHandler
from classes.schrock_paralell import Schrock
from utils.sa import SaScorer


class GeneticAlgorithm(GA):
    """Genetic algorithm based on GA base class.

    Args:
        args(dict): Dictionary containing all the commandline input args.
    """

    def __init__(
        self,
        args,
        mol_options: MoleculeOptions,
    ) -> None:
        self.args = args
        self.data_loader = DataLoader(args, mol_options)
        self.output_handler = OutputHandler(args)
        self.sascorer = SaScorer()
        self.submitit = SubmititHandler(args)
        self.switch_rate = 0.2
        super().__init__(
            mol_options=mol_options,
            population_size=args["population_size"],
            db_location=self.args["output_dir"]
            / f"ga_{time.strftime('%Y-%m-%d_%H-%M')}.sqlite",
            maximize_score=args["minimize_score"],
        )

    def make_initial_population(self) -> List[Schrock]:
        "Utility function to load starting list of catalysts"
        return self.data_loader.load_data()

    def _get_ligand_distribution(
        self,
        ligands: List[
            Union[
                catalystGA.components.CovalentLigand, catalystGA.components.DativeLigand
            ]
        ],
    ) -> Dict[str, List[int]]:
        """Helper function that takes list of ligands and returns a dict with
        the idx of the type of ligands."""

        ids = {"CovalentLigands": [], "DativeLigands": [], "BidentateLigands": []}
        for i, elem in enumerate(ligands):
            if isinstance(elem, CovalentLigand):
                ids["CovalentLigands"].append(i)
            elif isinstance(elem, DativeLigand):
                ids["DativeLigands"].append(i)
            elif isinstance(elem, BidentateLigand):
                ids["BidentateLigands"].append(i)
        return ids

    def crossover(self, ind1: Schrock, ind2: Schrock) -> Schrock or None:
        """Crossover for two catalysts.

        Returns None if no valid molecules is found.
        """
        # Get the ligand class type of ind1
        ind_type = type(ind1)

        # Choose one ligand at random from ind1 and crossover with same type of ligand from ind2, then replace this ligand in ind1 with the child
        new_mol = None
        counter = 0
        while not new_mol:
            inds = [ind1, ind2]
            selected_candidate = inds.pop(random.randint(0, 1))
            ind_ligands = copy.deepcopy(selected_candidate.ligands)

            # Get ligand type distribution in ligand list
            ids = self._get_ligand_distribution(ind_ligands)
            valid_ids = [
                ids[key] for key in ids.keys() if key != "BidentateLigands" and ids[key]
            ][0]

            idx = random.choice(valid_ids)
            # Get ligand type of  selected ligand
            ligand_type = type(selected_candidate.ligands[idx])

            # Allow for simply combining ligands between the two individuals
            if random.random() <= self.switch_rate:
                # Select the remaining molecule in the inds list and take one of its ligands.
                new_mol = inds[0].ligands[idx].mol
            else:
                # Crossover the same type of ligand in both catalysts
                new_mol = graph_crossover(ind1.ligands[idx].mol, ind2.ligands[idx].mol)
            counter += 1

            # If we have tried 10 times, return None
            if counter > 10:
                return None

        # This try/except will catch if new_mol has no donor atom
        try:
            Chem.SanitizeMol(new_mol)  # Extra sanitize, why not
            new_ligand = ligand_type(new_mol)

            # Use ligand type to create the right ligand distribution in new mol
            if isinstance(new_ligand, CovalentLigand):
                if len(ids["BidentateLigands"]) > 0:
                    ind_ligands = [new_ligand] * len(ids["CovalentLigands"]) + [
                        ind_ligands[x] for x in ids["BidentateLigands"]
                    ]
                else:
                    ind_ligands = [new_ligand] * len(ids["CovalentLigands"]) + [
                        ind_ligands[x] for x in ids["DativeLigands"]
                    ]
            elif isinstance(new_ligand, DativeLigand):
                ind_ligands = [ind_ligands[x] for x in ids["CovalentLigands"]] + [
                    new_ligand
                ] * len(ids["DativeLigands"])

            # Instantiate new catalyst and assemble the mol object. Ensures that an error is thrown here if the catalyst is invalid.
            child = ind_type(ind1.metal, ind_ligands)
            child.assemble()
            return child
        except Exception as e:
            print(f"Error in crossover with traceback: {e}")
            return None

    def mutate(self, ind: Schrock) -> Schrock or None:
        """Mutate the graph of one ligand of a Schrock catalyst."""
        #
        # Get ligand list type idxs
        ids = self._get_ligand_distribution(ind.ligands)
        valid_ids = [
            ids[key] for key in ids.keys() if key != "BidentateLigands" and ids[key]
        ][0]

        # pick one ligand at random
        idx = random.choice(valid_ids)
        # Get the type of the chosen ligand.
        ligand_type = type(ind.ligands[idx])

        new_mol = None
        counter = 0
        # Perform mutation operation on chosen ligand.
        while not new_mol:
            new_mol = graph_mutate(ind.ligands[idx].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            new_ligand = ligand_type(new_mol)

            # Create ligands list based on the format of the input list.
            # Use ligand type to create the right ligand distribution in new mol
            if isinstance(new_ligand, CovalentLigand):
                if len(ids["BidentateLigands"]) > 0:
                    ind.ligands = [new_ligand] * len(ids["CovalentLigands"]) + [
                        ind.ligands[x] for x in ids["BidentateLigands"]
                    ]
                else:
                    ind.ligands = [new_ligand] * len(ids["CovalentLigands"]) + [
                        ind.ligands[x] for x in ids["DativeLigands"]
                    ]
            elif isinstance(new_ligand, DativeLigand):
                ind.ligands = [ind.ligands[x] for x in ids["CovalentLigands"]] + [
                    new_ligand
                ] * len(ids["DativeLigands"])

            ind.assemble()  # Assemble to ensure that an invalid catalyst is not passed on from here.
            return ind
        except Exception as e:
            print(f"Error in mutation with traceback: {e}")
            return None

    def calculate_scores(self, population: List[Schrock], gen_id: int) -> List[Schrock]:
        """Helper function to call the submitit handler."""
        population = self.submitit.calculate_scores(population, gen_id)
        return population

    @staticmethod
    def _wrap_find_donor_atoms(
        individual, smarts_match, reference_smiles, n_cores, envvar_scratch
    ):
        """Calculate and the set connection atom ids for the ligands in a
        catalyst."""
        _logger = logging.getLogger("ligand")

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        )
        _logger.addHandler(stream)

        # Setup scrach directory
        scratch = Path(envvar_scratch)
        os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
        #
        # We only need to calculate once for all unique ligands.
        unique = set(individual.ligands)

        for i, ligand in enumerate(unique):
            calc_dir = scratch / f"{individual.idx[0]:03d}_{individual.idx[1]:03d}_{i}"
            calc_dir.mkdir(exist_ok=True)

            ligand.find_donor_atom(smarts_match, reference_smiles, n_cores, calc_dir)

            # Loop the ligand list in "individual" and set the id on matching ligand types.
            for elem in individual.ligands:
                if type(elem) == type(ligand):
                    elem.connection_atom_id = ligand.connection_atom_id
            _logger.info(
                f"After binding check the donor id is: Atom: {ligand.mol.GetAtomWithIdx(ligand.connection_atom_id).GetSymbol()} Idx: {ligand.connection_atom_id} for {ligand}"
            )

        return [ligand.connection_atom_id for ligand in individual.ligands]

    def run(self):
        "Custom runner function for the GA"

        # Create the initial population of Schrock object catalysts.
        self.population = self.make_initial_population()

        logging.info("Getting scores for starting population")
        self.population = self.calculate_scores(self.population, gen_id=0)
        self.sort_population(self.population, self.maximize_score)

        # Functionality to check synthetic accessibility
        if self.args["sa_screening"]:
            self.sascorer.get_sa(self.population)

        # Normalize the score of population individuals to value between 0 and 1
        self.calculate_fitness(self.population)

        # Save the generation as pickle file
        self.output_handler.save(
            self.population,
            directory=self.args["output_dir"] / "pickles",
            name="GA00.pkl",
        )

        # Write the current population attributes to file
        self.output_handler.write_out(self.population, name="GA.out", genid=0)

        logging.info("Finished initial generation")

        # Start evolving
        generation_num = 0
        for generation in range(self.args["generations"]):
            # Counter for tracking generation number
            generation_num += 1
            logging.info("Starting generation %d", generation_num)

            # If debugging simply reuse previous population
            if self.args["debug"]:
                children = self.reproduce(self.population, genid=generation_num)
            else:
                children = self.reproduce(self.population, genid=generation_num)

            # Calculate new scores
            logging.info(f"Getting scores for gen {generation_num}")
            children = self.calculate_scores(children, gen_id=generation_num)

            # Functionality to compute synthetic accessibility
            if self.args["sa_screening"]:
                logging.info("Getting synthetic accessibility score")
                self.sascorer.get_sa(children)

            # Save the population list, used for debugging
            self.output_handler.save(
                children,
                directory=self.args["output_dir"] / "pickles",
                name=f"GA{generation_num:02d}_debug2.pkl",
            )

            # Combine the previous population with the new children and the best
            self.population = self.prune(self.population + children)

            # Normalize new scores to prep for next gen
            self.calculate_fitness(self.population)

            # Sort children for print out. Then print to separate log file for children
            self.sort_population(children, self.maximize_score)
            self.output_handler.write_out(
                children,
                name="GA_children.out",
                genid=generation_num,
            )

            self.output_handler.write_out(
                self.population, name="GA.out", genid=generation_num
            )

            # Save current population
            self.output_handler.save(
                self.population,
                directory=self.args["output_dir"] / "pickles",
                name=f"GA{generation_num:02d}.pkl",
            )

            logging.info(f"Gen No. {generation_num} finished")

        logging.info("All generation finished. Uploading final pickle to ERDA")
