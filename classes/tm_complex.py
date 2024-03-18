import copy
import logging
import math
import os
import pickle
import random
import socket
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from catalystGA import BaseCatalyst
from catalystGA.xtb import ac2mol
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDetermineBonds, rdDistGeom

_logger = logging.getLogger(__name__)

hartree2kcalmol = 627.5094740631


def _determineConnectivity(mol, **kwargs):
    """Determine bonds in molecule."""
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


class Tm_Complex(BaseCatalyst):
    """Class to represent metal + ligand."""

    def __init__(self, metal: Chem.Mol, ligands: List):
        super().__init__(metal, ligands)

    @property
    def dispatcher(self):
        return {
            "calculate_score": self.calculate_score,
        }

    @staticmethod
    def adjacency_check(adj1, atoms, coords) -> Any:
        "For detecting broken bonds during optimizations"
        mol = ac2mol(atoms, coords)
        _determineConnectivity(mol, useHueckel=True)
        adj2 = Chem.GetAdjacencyMatrix(mol, force=True)
        return np.array_equal(adj1, adj2)

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
        embedding the NxHx moieties when creating the catalytic cycle.

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

        # Add Extra Ligands
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

    def calculate_score(self, args=None) -> None:
        "Helper function to do debugging of the GA workflow with a simple score"

        start = time.time()

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        str = logging.StreamHandler()
        str.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        )
        _logger.addHandler(str)
        _logger.warning(socket.gethostname())

        _logger.info("Performing whaterver calculation on tm_complex")
        # Add some delay to test wait functionality
        if random.uniform(0, 1) > 0.5:
            time.sleep(10)

        self.score = random.uniform(0, 1)
        self.timing = time.time() - start
        self.save()

        _logger.info("Calc finished.")

    def save(self, directory=".") -> None:
        """Dump ind object into file."""
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
        return vars(self)

    def clean(self) -> None:
        "Utility method to reset attributes"
        self.score = math.nan
        self.embedded_mols = None
        self.energy = math.nan
        self.error = None
        self.fitness = math.nan
        self.timing = math.nan
        self.results_dict = None
        return


if __name__ == "__main__":
    print("Nothing here")
