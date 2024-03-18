"""Module that contains mol manipulations and various reusable
functionality."""
import copy
import logging
import os
import subprocess
import sys

import numpy as np
import io
import pickle
from rdkit import Chem


class ConditionalFormatter(logging.Formatter):
    """Utility function to chance the formatting of logging output with a
    simple keyword on the logger instance."""

    def format(self, record):
        if hasattr(record, "simple") and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


def load_gen_file(path=None):
    with open(path, "rb") as f:
        gen = renamed_load(f)
    return gen


def remove_N2(mol):
    """Remove N2 group on mol."""
    # Substructure match the N2
    NH2_match = Chem.MolFromSmarts("N#N")
    removed_mol = Chem.DeleteSubstructs(mol, NH2_match)

    return removed_mol


def mol_with_atom_index(mol):
    """Visualize mol object with atom indices."""
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    Chem.Draw.MolToImage(mol, size=(900, 900)).show()
    return mol


def sdf2mol(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)[0]
    return mol


def mols_from_smi_file(smi_file, n_mols=None):
    mols = []
    with open(smi_file) as _file:
        for i, line in enumerate(_file):
            mols.append(Chem.MolFromSmiles(line))
            if n_mols:
                if i == n_mols - 1:
                    break
    return mols


def get_git_revision_short_hash() -> str:
    """Get the git hash of current commit if git repo."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def energy_filter(confs, energies, optimized_mol, scoring_args):
    """Filter out higher energy conformers based on energy cutoff.

    Args:
        confs: Sequnce of conformers objects.
        energies (List): List of conformer energies
        optimized_mol (Chem.Mol): Optimized mol object
        scoring_args: SubmitIt function arg dict

    Returns:
        energies: Filtered energies
        new_mol: Mol object with filtered conformers
    """
    mask = energies < (energies.min() + scoring_args["energy_cutoff"])
    print(mask, energies)
    confs = list(np.array(confs)[mask])
    new_mol = copy.deepcopy(optimized_mol)
    new_mol.RemoveAllConformers()
    for c in confs:
        new_mol.AddConformer(c, assignId=True)
    energies = energies[mask]

    return energies, new_mol


def shell(args, shell=False):
    """Subprocess handler function where output is stored in files.

    Args:
        cmd (str): String to pass to bash shell
        shell (bool): Specifies whether run as bash shell or not

    Returns:
        output (str): Program output
        err (str): Possible error messages
    """
    cmd, key = args
    print(f"String passed to shell: {cmd}")
    if shell:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    else:
        cmd = cmd.split()
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    output, err = p.communicate()

    with open(f"{key}job.out", "w") as f:
        f.write(output)
    with open(f"{key}err.out", "w") as f:
        f.write(err)

    return output, err


class cd:
    """Context manager for changing the current working directory dynamically.

    # See:
    https://book.pythontips.com/en/latest/context_managers.html
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        # Print traceback if anything happens
        if traceback:
            print(sys.exc_info())
        os.chdir(self.savedPath)


def read_file(file_name):
    """Read smiles from file and return mol generator."""
    with open(file_name, "r") as file:
        for smiles in file:
            yield Chem.MolFromSmiles(smiles)


def read_file_tolist(file_name):
    mol_list = []
    with open(file_name, "r") as file:
        for smiles in file:
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


def write_xtb_constrain_file(mol, constrain_idx, path):
    idxs = []
    for elem in mol.GetAtoms():
        idxs.append(elem.GetIdx() + 1)
    match = [idx for idx in idxs if idx not in constrain_idx]

    return match


def catch(func, *args, handle=lambda e: e, **kwargs):
    """Helper function that takes the submitit result and returns an exception
    if no results can be retrieved."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print("Error in catch function. Traceback : {e}")
        return handle(e)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "my_utils.classes":
            renamed_module = "utils.classes"
        elif module == "my_utils.classes":
            renamed_module = "scripts.classes"
        if module == "catalystGA.components_paralell":
            renamed_module = "catalystGA.components"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
