"""Module for the Schrock class."""
import copy
import logging
import math
import os
import pickle
import random
import socket
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from catalystGA import BaseCatalyst
from catalystGA.components import BidentateLigand, CovalentLigand, Metal
from catalystGA.xtb import ac2mol
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDetermineBonds, rdDistGeom

from .XTB_class import XTB_optimize_schrock

_logger = logging.getLogger(__name__)

# Conversion factor for Hartree -> kcal/mol
hartree2kcalmol = 627.5094740631

# Chiral tag mapper
chiral_tags = {
    "4": Chem.CHI_TRIGONALBIPYRAMIDAL,
    "3": Chem.CHI_TETRAHEDRAL,
    "5": Chem.CHI_OCTAHEDRAL,
}

# Permuration order mapper
permutation_checks = {"5": [1, 6, 12], "4": [1, 6], "3": [1]}


@hide_warnings
def _determineConnectivity(mol, **kwargs):
    """Determine connectivity in molecule.

    Use to check if bonds are broken in xTB optimizations
    """
    try:
        rdDetermineBonds.DetermineConnectivity(mol, **kwargs)
    finally:
        # cleanup extended hueckel files
        try:
            os.remove("nul")
            os.remove("run.out")
        except FileNotFoundError:
            pass
    return mol


class Schrock(BaseCatalyst):
    """Class to represent the catalyst structures."""

    def __init__(self, metal: Chem.Mol, ligands: List):
        """Ligands is a list that contain 3 CovalentLigands and either 1 or 2
        DativeLigands."""
        super().__init__(metal, ligands)

    @property
    def dispatcher(self):
        "Enable calling of different scoring functions easily"
        return {
            "calculate_score": self.calculate_score,
            "calculate_score_logP": self.calculate_score_logP,
        }

    @staticmethod
    def adjacency_check(adj1, atoms, coords) -> Any:
        "For detecting broken bonds during optimizations"
        mol = ac2mol(atoms, coords)
        _determineConnectivity(mol, useHueckel=True)
        adj2 = Chem.GetAdjacencyMatrix(mol, force=True)
        return np.array_equal(adj1, adj2)

    def calculate_score_logP(self, args=None) -> None:
        "Helper function to do debugging of the GA workflow with a simple scoring function"

        start = time.time()

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        str = logging.StreamHandler()
        str.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        )
        _logger.addHandler(str)
        _logger.warning(socket.gethostname())

        # Add some delay to test wait functionality submitit
        if random.uniform(0, 1) > 0.5:
            time.sleep(10)

        self.score = random.uniform(0, 1)
        self.timing = time.time() - start
        self.save()
        _logger.info("LogP score finalized.")

    def calculate_score(self, args=None) -> None:
        """Calculate score for the catalyst."""
        scratch = args["output_dir"] / args["scratch"]
        scratch.mkdir(parents=True, exist_ok=True)

        start = time.time()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.StreamHandler(),  # For debugging. Can be removed on remote
            ],
        )
        _logger.warning(socket.gethostname())
        _logger.info(f"Calculating score for {self}\nSMILES: {self.smiles}\n")

        # Depending on the number of ligands, we need to test different permutations
        if any(isinstance(x, BidentateLigand) for x in self.ligands):
            num_sites = str(len(self.ligands) + 1)
        else:
            num_sites = str(len(self.ligands))

        embedded_structures = {}
        options_params = {
            "Mo_N2": {"charge": 0, "uhf": 1},
            "Mo_NNH+": {"charge": 1, "uhf": 1},
            "Mo_NNH": {"charge": 0, "uhf": 0},
        }
        results_dict = defaultdict(dict)

        # Set the timeout limit for the xtb calc. Assign each intermediate their own share of the
        # total time
        args["timeout_min"] = (
            args["timeout_min"] // len(options_params) - 1
        )  # Substrract 1 minute buffer
        _logger.info(f"Timeout per xtb calc is set to : {args['timeout_min']}")

        # Create moles with different permutations of the ligands.
        permutation_mols = []
        for permutationOrder in permutation_checks[num_sites]:
            permutation_mols.append(
                self.embed(
                    extraLigands="[Mo:1]>>[Mo:1](<-N#N)",
                    chiralTag=chiral_tags[num_sites],
                    permutationOrder=permutationOrder,
                    numConfs=args["n_confs"],
                    useRandomCoords=True,
                    pruneRmsThresh=args["rms_prune"],
                    numThreads=args["cpus_per_mol"],
                )
            )

        # Get binding energies for the different permutations
        energies = []
        for i, permutation in enumerate(permutation_mols):
            _logger.info(f"Checking permutation {Chem.MolToSmiles(permutation)}")

            # Instantiate optimizer class
            tempdir = (
                scratch
                / "permutation"
                / f"{self.idx[0]:03d}_{self.idx[1]:03d}_Mo_N2_{i}"
            )
            tempdir.mkdir(parents=True, exist_ok=True)
            args["name"] = tempdir
            args["charge"] = options_params["Mo_N2"]["charge"]
            args["uhf"] = options_params["Mo_N2"]["uhf"]
            optimizer = XTB_optimize_schrock(mol=permutation, scoring_options=args)
            energy_opt = optimizer.optimize_permutation()

            if len(energy_opt) >= 1:
                energies.append(np.min(energy_opt))
            else:
                _logger.warning(
                    f"OBS! Permutation check failed. No valid conformers for {self.smiles}"
                )
                return

        # Get lowest energy permutation. If none found, use first permutation and continue.
        if len(energies) >= 1:
            min_idx = min(range(len(energies)), key=energies.__getitem__)
        else:
            min_idx = 0

        # Select the lowest energy permutation
        permutationOrder = permutation_checks[num_sites][min_idx]
        embedded_structures["Mo_N2"] = permutation_mols[min_idx]

        _logger.info("Done with permutation check. Starting scoring loop")
        for elem, extraligand in zip(
            options_params.keys(),
            ["[Mo:1]>>[Mo:1](<-N#N)", "[Mo:1]>>[Mo:1](-N=N)", "[Mo:1]>>[Mo:1](-N=N)"],
        ):
            # Pass on N2 as this was aready obtained in permutation check
            if elem == "Mo_N2":
                continue

            embedded_structures[elem] = self.embed(
                extraLigands=extraligand,
                chiralTag=chiral_tags[num_sites],
                permutationOrder=permutationOrder,
                numConfs=args["n_confs"],
                useRandomCoords=True,
                pruneRmsThresh=args["rms_prune"],
                numThreads=args["cpus_per_mol"],
            )

            # Check if embedding succeded.
            n_confs = embedded_structures[elem].GetConformers()
            if len(n_confs) == 0:
                _logger.warning(f"Embedding failed for {elem}. Returning.")
                return
            else:
                _logger.info(
                    f'Number of conformers asked for : {args["n_confs"]}, Numbers left after embedding with pruning: {len(n_confs)}'
                )

        # Get energies for all embedded intermediates.
        for key, mol in embedded_structures.items():
            _logger.info(f"Starting embedding optimization of {key}")

            calc_dir = (
                scratch / f"{self.idx[0]:03d}_{self.idx[1]:03d}_{key}_{uuid.uuid4()}"
            )

            # Determine connectivity for mol
            mol_adj = Chem.GetAdjacencyMatrix(mol)

            mol_atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            _logger.info(f"Doing: {key} at {calc_dir}")

            # Instantiate optimizer class
            args["name"] = calc_dir
            args["charge"] = options_params[key]["charge"]
            args["uhf"] = options_params[key]["uhf"]
            optimizer = XTB_optimize_schrock(mol=mol, scoring_options=args)
            optimized_mol, energies = optimizer.optimize_schrock()

            _logger.info(f"Number of converged energies found: {len(energies)}")
            if len(energies) == 0:
                _logger.info("Returning since no energies found.")
                return

            # Remove conformers that are too far from the minimum energy conf and remove conformers that have broken bonds
            energies, filtered_mol = self.energy_filter(
                energies, optimized_mol, args, mol_atoms, mol_adj
            )

            if len(filtered_mol.GetConformers()) == 0:
                _logger.warning("No valid conformers found. Returning from calculation")
                return
            else:
                _logger.info(
                    f"Number of valid structures found after energy filtering and bond check: {len(energies)}"
                )
                results_dict[key]["optimized_mol"] = filtered_mol
                results_dict[key]["energy"] = np.min(energies)
                results_dict[key]["energies"] = energies

        # Save results
        self.embedded_mols = embedded_structures
        self.results_dict = results_dict

        try:
            # Get the normalized score
            energy_score1, energy_score2 = self.get_energies(results_dict)
            mean = (energy_score1 + energy_score2) / 2

            self.score = self.get_normalized_score(
                (energy_score1, energy_score2), maximize_score=args["minimize_score"]
            )
            self.energy = mean

        except Exception as e:
            _logger.info(f"Score calculation failed for {self.idx}. Traceback : {e}")
        self.timing = time.time() - start

    def get_energies(self, results_dict) -> Tuple[Any, Any]:
        """Get the score for the protonation and reduction reaction hardcoded
        energies are from gfn2 xTB calculations."""

        Lu_energy = -22.505650727184
        LuH_plus_energy = -22.679462932969
        CrCp_energy = -60.159896098345
        CrCp_plus_energy = -59.841744813561
        energy_score1 = (
            results_dict["Mo_NNH+"]["energy"]
            + Lu_energy
            - (results_dict["Mo_N2"]["energy"] + LuH_plus_energy)
        ) * hartree2kcalmol
        energy_score2 = (
            results_dict["Mo_NNH"]["energy"]
            + CrCp_plus_energy
            - (results_dict["Mo_NNH+"]["energy"] + CrCp_energy)
        ) * hartree2kcalmol
        return energy_score1, energy_score2

    def get_normalized_score(
        self, energies, maximize_score=True, newRange=(0, 1)
    ) -> Any:
        "Normalize reaction energies to the range 0-1"

        # These xmin and xmax values are hardcoded based on expected good and bad values given the scoring function.
        if maximize_score:
            xmin, xmax = 100, -100
        else:
            xmin, xmax = -100, 100

        norms = []
        for energy in energies:
            if math.isnan(energy):
                norm = 0
            else:
                norm = (energy - xmin) / (xmax - xmin)  # scale between zero and one
            if newRange == (0, 1):
                pass
            elif newRange != (0, 1):
                norm = (
                    norm * (newRange[1] - newRange[0]) + newRange[0]
                )  # scale to a different range.

            if norm > 1:
                norm = 1
            elif norm < 0:
                norm = 0

            norms.append(norm)

        # Multiply the normalized energies
        product = 1
        for num in norms:
            product *= num

        return product

    def save(self, directory=".") -> None:
        """Save catalyst object into file."""
        filename = os.path.join(directory, "../ind.pkl")
        with open(filename, "wb+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def energy_filter(
        self, energies, optimized_mol, scoring_args, mol_atoms, mol_adj
    ) -> Tuple[list]:
        """Filter out higher energy conformers based on energy cutoff and bond
        check.

        Args:
            confs: Sequnce of conformers objects.
            energies (List): List of conformer energies
            optimized_mol (Chem.Mol): Optimized mol object
            scoring_args: Scoring function arg dict

        Returns:
            energies: Filtered energies
            new_mol: Mol object with filtered conformers
        """
        confs = optimized_mol.GetConformers()
        mask = energies < (energies.min() + scoring_args["energy_cutoff"])
        confs = list(np.array(confs)[mask])
        new_mol = copy.deepcopy(optimized_mol)
        new_mol.RemoveAllConformers()

        energies_filtered = energies[mask]

        _logger.info(
            f"Number of energies filtered out with cutoff of {scoring_args['energy_cutoff']}: {len(energies)-len(energies_filtered)}"
        )

        filtered_energy = []
        for conf, energy in zip(confs, energies_filtered):
            mol_opt_coords = conf.GetPositions()
            # Check adjacency matrix
            if not self.adjacency_check(mol_adj, mol_atoms, mol_opt_coords):
                _logger.warning(
                    "Change in adjacency matrix after pre-optimization . Non valid struct."
                )
                continue
            else:
                new_mol.AddConformer(conf, assignId=True)
                filtered_energy.append(energy)

        return filtered_energy, new_mol

    def get_props(self) -> Dict[str, Any]:
        "Utility function to return class attributes"
        return vars(self)

    def clean(self) -> None:
        "Utility method to reset attributes on catalyst instance"
        self.score = math.nan
        self.embedded_mols = None
        self.energy = math.nan
        self.error = None
        self.fitness = math.nan
        self.timing = math.nan
        self.results_dict = None
        return

    def embed_constrained_dft(
        self,
        core_mol=None,
        extraLigands=None,
        chiralTag=None,
        permutationOrder=None,
        numConfs=10,
        useRandomCoords=True,
        pruneRmsThresh=-1,
        **kwargs,
    ):
        """Embed the Catalyst Molecule using ETKDG. Specific method used for
        embedding the NxHx moieties when creating the catalytic cycle. Not
        relevant to the GA.

        Args:
            core_mol (Chem.Mol): Mol object of the core catalyst structure without any NxHx moieties.
            extraLigands (str, optional): Reaction SMARTS to add ligands to the molecule. Defaults to None.
            chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom. Defaults to None.
            permutationOrder (int, optional): Permutation order of ligands. Defaults to None.
            numConfs (int, optional): Number of Conformers to embed. Defaults to 10.
            useRandomCoords (bool, optional): Embedding option. Defaults to True.
            pruneRmsThresh (int, optional): Conformers within this threshold will be removed. Defaults to -1.

        Returns:
            Chem.Mol: Catalyst Molecule with conformers embedded
        """

        # Add extra ligands to Mo core
        if extraLigands:
            rxn = rdChemReactions.ReactionFromSmarts(extraLigands)
            tmp = rxn.RunReactants((core_mol,))[0][0]

        # Add hydrogens
        Chem.SanitizeMol(tmp)
        mol3d = Chem.AddHs(tmp)

        # Match the core+ligand to the Mo core.
        core_mol = Chem.AddHs(core_mol)
        match = mol3d.GetSubstructMatch(core_mol)
        if not match:
            raise ValueError("molecule doesn't match the core")

        # Get the coordinates for the core, which constrains the embedding
        coreConf = core_mol.GetConformer()
        coordMap = {}
        for i, idxI in enumerate(match):
            corePtI = coreConf.GetAtomPosition(i)
            coordMap[idxI] = corePtI

        metal = mol3d.GetAtomWithIdx(mol3d.GetSubstructMatch(self.metal.atom)[0])
        self._setChiralTagAndOrder(metal, chiralTag, permutationOrder)

        # Embed with ETKDG
        _ = rdDistGeom.EmbedMultipleConfs(
            mol3d,
            coordMap=coordMap,
            numConfs=numConfs,
            useRandomCoords=True,  # For making these cycle this NEEDS TO BE TRUE! OTHERWISE COORDINATES IN THE CORE CHANGE
            pruneRmsThresh=pruneRmsThresh,
            **kwargs,
        )
        return mol3d


if __name__ == "__main__":
    metal: Any = Metal("Mo")
    ligands: list = [CovalentLigand(Chem.MolFromSmiles("CCCC"))] * 3 + [
        BidentateLigand(Chem.MolFromSmiles("c1cnc2c(c1)ccc1cccnc12"))
    ]

    sch: Schrock = Schrock(metal, ligands)
    tmp: Any = sch.assemble_bidentate()
