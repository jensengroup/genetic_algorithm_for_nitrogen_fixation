"""Module for xtb functionality."""

import concurrent.futures
import copy
import logging
import math
import os
import random
import re
import shutil
import string
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from catalystGA.xtb import set_threads
from rdkit import Chem
from rdkit.Geometry import Point3D

_logger = logging.getLogger("xtb")


def run_xtb(args):
    """Submit xtb calculations with given params.

    Args:
        args (tuple): runner parameters

    Returns:
        results (tuple): Energy and geometries of calculated structures
    """
    xyz_file = args["xyz_file"]
    xtb_cmd = args["xtb_cmd"]
    numThreads = args["numThreads"]
    conf_path = args["conf_path"]
    logname = args["logname"]
    timeout = args["timeout_min"]
    _logger.info(
        f"running {xyz_file} on {numThreads} core(s) starting at {datetime.now()} with timeout: {timeout} min"
    )

    cwd = os.path.dirname(xyz_file)
    xyz_file = os.path.basename(xyz_file)
    cmd = f"{xtb_cmd} -- {xyz_file} "

    set_threads(numThreads)

    popen = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=False,
        cwd=cwd,
    )

    try:
        output, err = popen.communicate(timeout=timeout * 60)
    except subprocess.TimeoutExpired:
        _logger.warning("Timeout expired")
        popen.kill()
        output, err = popen.communicate()

    # Save xtb output files
    with open(Path(conf_path) / f"{logname}_job.out", "w") as f:
        f.write(output)
    with open(Path(conf_path) / f"{logname}_err.out", "w") as f:
        f.write(err)

    # Get results from output
    results = read_results(output, err)
    if not results["energy"]:
        _logger.warning(f"No xTB energy returned for {conf_path}")
        # _logger.warning(output)
        # _logger.warning(err)

    return results


def extract_energyxtb(logfile=None):
    """Extracts xtb energies from xtb logfile using regex matching.

    Args:
        logfile (str): Specifies logfile to pull energy from

    Returns:
        energy (List[float]): List of floats containing the energy in each step
    """
    re_energy = re.compile("energy: (-\\d+\\.\\d+)")
    energy = []
    with logfile.open() as f:
        for line in f:
            if "energy" in line:
                energy.append(float(re_energy.search(line).groups()[0]))
    return energy


def read_results(output, err):
    """Get coordinates and energy from xtb output."""
    if "normal termination" not in err:
        return {"atoms": None, "coords": None, "energy": None}
    lines = output.splitlines()
    energy = None
    structure_block = False
    atoms = []
    coords = []
    for line in lines:
        if "final structure" in line:
            structure_block = True
        elif structure_block:
            s = line.split()
            if len(s) == 4:
                atoms.append(s[0])
                coords.append(list(map(float, s[1:])))
            elif len(s) == 0:
                structure_block = False
        elif "TOTAL ENERGY" in line:
            energy = float(line.split()[3])
    return {"atoms": atoms, "coords": coords, "energy": energy}


def write_xyz(atoms, coords, destination_dir):
    """Write .xyz file from atoms and coords."""
    file = destination_dir / "mol.xyz"
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(file, "w") as inp:
        inp.write(xyz)

    return file


class XTB_optimizer:
    """Base XTB optimizer class."""

    def __init__(self):
        # Initialize default xtb values
        self.method = "ff"
        self.workers = 1
        # xtb runner function
        self.run_xtb = run_xtb
        # xtb options
        self.XTB_OPTIONS = {}

        # Get default cmd string
        cmd = f"xtb --gfn{self.method}"
        for key, value in self.XTB_OPTIONS.items():
            cmd += f" --{key} {value}"
        self.cmd = cmd

    def add_options_to_cmd(self, option_dict):
        """From passed dict get xtb options if it has the appropriate keys and
        add to xtb string command."""

        # XTB options to check for
        options = ["gbsa", "spin", "charge", "uhf", "input", "opt"]

        # Get commandline options
        commands = {k: v for k, v in option_dict.items() if k in options}
        for key, value in commands.items():
            self.cmd += f" --{key} {value}"

    def optimize(self, args):
        """Do paralell optimization of all the entries in args."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:
            results = executor.map(self.run_xtb, args)
        return results

    @staticmethod
    def _write_xtb_input_files(fragment, name, destination="."):
        """Utility method to write xyz input files from mol object."""
        number_of_atoms = fragment.GetNumAtoms()
        symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
        conformers = fragment.GetConformers()
        file_paths = []
        conf_paths = []
        for i, conf in enumerate(conformers):
            conf_path = os.path.join(destination, f"conf{i:03d}")
            conf_paths.append(conf_path)

            if os.path.exists(conf_path):
                shutil.rmtree(conf_path)
            os.makedirs(conf_path)

            file_name = f"{name}{i:03d}.xyz"
            file_path = os.path.join(conf_path, file_name)
            with open(file_path, "w") as _file:
                _file.write(str(number_of_atoms) + "\n")
                _file.write(f"{Chem.MolToSmiles(fragment)}\n")
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    _file.write(line)
            file_paths.append(file_path)
        return file_paths, conf_paths


class XTB_optimize_schrock(XTB_optimizer):
    """Specific xtb optimizer class for the schrock intermediates."""

    def __init__(self, mol, scoring_options, **kwargs):
        """

        Args:
            mol (Chem.rdchem.Mol): Mol object to score
            scoring_options (dict): Scoring options for xtb
        """
        # Inherit the basic xtb functionality from XTB_OPTIMIZER
        super().__init__(**kwargs)

        # Set class attributes
        self.mol = mol
        self.options = scoring_options

        # Set additional xtb options
        self.add_options_to_cmd(self.options)

        # Set folder name if given options dict
        if "name" not in self.options:
            self.name = "tmp_" + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=4)
            )
        else:
            self.name = self.options["name"]

    @staticmethod
    def copy_logfile(conf_paths, name="opt.log"):
        """Copy xtbopt.log file to new name."""
        for elem in conf_paths:
            shutil.copy(os.path.join(elem, "xtbopt.log"), os.path.join(elem, name))

    def optimize_schrock(self):
        """Optimize the given mol object.

        Returns:
            mol_opt: optimized mol object with all the conformers
        """
        # Write input files
        xyz_files, conf_paths = self._write_xtb_input_files(
            self.mol, "xtbmol", destination=self.name
        )

        self.workers = np.min([self.options["cpus_per_mol"], self.options["n_confs"]])
        cpus_per_worker = self.options["cpus_per_mol"] // self.workers
        _logger.info(f"workers: {self.workers}, cpus_per_worker: {cpus_per_worker}")
        self.cmd = self.cmd + f" --parallel {cpus_per_worker}"
        # Create args tuple and submit ff calculation
        args = [
            (
                {
                    "xyz_file": xyz_file,
                    "xtb_cmd": self.cmd,
                    "numThreads": cpus_per_worker,
                    "conf_path": conf_paths[i],
                    "logname": "ff",
                    "timeout_min": self.options["timeout_min"],
                }
            )
            for i, xyz_file in enumerate(xyz_files)
        ]
        result = self.optimize(args)

        # Store the log file under given name
        # self.copy_logfile(conf_paths, name="ffopt.log")

        # Change from ff to given method
        self.cmd = self.cmd.replace("gfnff", f"gfn {self.options['method']}")

        # Get the new input files and args
        xyz_files = [Path(xyz_file).parent / "xtbopt.xyz" for xyz_file in xyz_files]

        args = [
            (
                {
                    "xyz_file": xyz_file,
                    "xtb_cmd": self.cmd,
                    "numThreads": cpus_per_worker,
                    "conf_path": conf_paths[i],
                    "logname": f"gfn{self.options['method']}",
                    "timeout_min": self.options["timeout_min"],
                }
            )
            for i, xyz_file in enumerate(xyz_files)
        ]
        # Optimize with current input constrain file.
        result = self.optimize(args)

        # Add optimized conformers to mol_opt
        mol_opt = copy.deepcopy(self.mol)
        mol_opt.RemoveAllConformers()

        energies = []
        # Add optimized conformers
        for i, res in enumerate(result):
            if res:
                if res["energy"]:
                    energies.append(res["energy"])
                    self._add_conformer2mol(
                        mol=mol_opt,
                        atoms=res["atoms"],
                        coords=res["coords"],
                    )
            else:
                _logger.info(f"Conformer {i} did not converge.")

        _logger.info(
            f"Finished all conformer optimizations with No. confs: {mol_opt.GetNumConformers()}"
        )

        # Clean up
        if self.options["cleanup"]:
            if mol_opt.GetNumConformers() == 0:
                pass
            else:
                shutil.rmtree(self.name)

        return mol_opt, np.array(energies)

    def optimize_permutation(self):
        """Optimize the given mol object.

        Returns:
            mol_opt: optimized mol object with all the conformers
        """
        # Write input files
        xyz_files, conf_paths = self._write_xtb_input_files(
            self.mol, "xtbmol", destination=self.name
        )

        self.workers = np.min([self.options["cpus_per_mol"], self.options["n_confs"]])
        cpus_per_worker = self.options["cpus_per_mol"] // self.workers

        _logger.info(f"workers: {self.workers}, cpus_per_worker: {cpus_per_worker}")
        self.cmd = self.cmd + f" --parallel {cpus_per_worker}"
        # Create args tuple and submit ff calculation
        args = [
            (
                {
                    "xyz_file": xyz_file,
                    "xtb_cmd": self.cmd,
                    "numThreads": cpus_per_worker,
                    "conf_path": conf_paths[i],
                    "logname": "ff",
                    "timeout_min": self.options["timeout_min"],
                }
            )
            for i, xyz_file in enumerate(xyz_files)
        ]
        result = self.optimize(args)

        # Store the log file under given name
        # self.copy_logfile(conf_paths, name="ffopt.log")

        # Change from ff to given method
        self.cmd = self.cmd.replace("gfnff", f"gfn {self.options['method']}")
        self.cmd = self.cmd.replace("--opt", "")
        self.cmd = self.cmd.replace("loose", "")
        self.cmd = self.cmd.replace("tight", "")

        # Get the new input files and args
        xyz_files = [Path(xyz_file).parent / "xtbopt.xyz" for xyz_file in xyz_files]

        args = [
            (
                {
                    "xyz_file": xyz_file,
                    "xtb_cmd": self.cmd,
                    "numThreads": cpus_per_worker,
                    "conf_path": conf_paths[i],
                    "logname": f"gfn{self.options['method']}",
                    "timeout_min": self.options["timeout_min"],
                }
            )
            for i, xyz_file in enumerate(xyz_files)
        ]
        # Optimize with current input constrain file.
        result = self.optimize(args)

        _logger.info("Finished all conformer optimizations")

        # Add optimized conformers to mol_opt
        mol_opt = copy.deepcopy(self.mol)
        mol_opt.RemoveAllConformers()

        energies = []
        # Add optimized conformers
        for i, res in enumerate(result):
            if res:
                if res["energy"]:
                    energies.append(res["energy"])
            else:
                energies.append(math.nan)
                _logger.info(f"Conformer {i} did not converge.")

        # Clean up
        if self.options["cleanup"]:
            if len(energies) == 0:
                pass
            else:
                shutil.rmtree(self.name)

        return np.array(energies)

    @staticmethod
    def _add_conformer2mol(mol, atoms, coords):
        """Add Conformer to rdkit.mol object."""
        conf = Chem.Conformer()
        for i in range(mol.GetNumAtoms()):
            # assert that same atom type
            assert (
                mol.GetAtomWithIdx(i).GetSymbol() == atoms[i]
            ), "Order of atoms is not the same in CREST output and rdkit Mol"
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)
