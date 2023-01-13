# -*- coding: utf-8 -*-
"""This module reads and scans amino-acid sequences for Pfam domains."""
from pathlib import Path
import subprocess
import pandas as pd  # type: ignore

from src.config import hmmscan, hmmpress  # type: ignore  # pylint: disable=import-error


def create_pfam_hmm_db(path: Path) -> Path:
    """Download PFAM hmm file and turn it into a database.

    Download PFAM hmm file for PFAM current release and
    turns it into a database that is ready for use for
    hmmscan. All files are stored in the folder at path.

    Parameters
    ----------
    path : pathlib.Path
        Path to the output folder

    Returns
    -------
    pathlib.Path
        Path to Pfam database
    """
    # Creates output folder if
    # it does not exist
    path.mkdir(exist_ok=True, parents=True)
    # Download Pfam hmm file
    subprocess.run(
        [
            "curl",
            "-o",
            path / "Pfam-A.hmm.gz",
            "http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz",
        ],
        check=True,
    )
    # Gunzip file
    subprocess.run(["gunzip", path / "Pfam-A.hmm.gz"], check=True)
    # Runs hmmpress to
    # prepare an HMM database
    # to run hmmscan
    subprocess.run([hmmpress, path / "Pfam-A.hmm"], check=True)
    return path / "Pfam-A.hmm"


def scan_for_pfam(
    path_to_fasta: Path,
    path_to_pfam_db: Path,
    path_to_output: Path,
    path_tmp: Path,
) -> Path:
    """Scan for PFAM domains in input amino acid sequences.

    Read sequences in fasta file at path_to_fasta, scans
    them for Pfam domains using Pfam HMM in the database
    at path_to_pfam_db. Filter out results to keep only
    domains for which e-value < 10^-5 and return them at
    csv format at path_to_output.

    Parameters
    ----------
    path_to_fasta : Path
        Path to input fasta file with amino-acid sequences
    path_to_pfam_db : Path
        Path to folder where Pfam HMM database is
    path_to_output : Path
        Path where output csv file will be written
    path_tmp : Path
        Path to temporary folder

    Returns
    -------
    Path
        Path to output csv file
    """
    # Creates temporary folder
    # if not exists
    path_tmp.mkdir(exist_ok=True, parents=True)
    # Retrieves input fasta file name
    # without extension
    filename = path_to_fasta.stem
    # Runs hmmscan
    subprocess.run(
        [
            hmmscan,
            "--tblout",
            path_tmp / f"{filename}.tsv",
            "--noali",
            path_to_pfam_db,
            path_to_fasta,
        ],
        check=True,
    )
    # Parses hmmscan output
    pfam_domain = list()
    seq_id = list()
    e_value = list()
    with open(path_tmp / f"{filename}.tsv", "r") as hmmscan_output:
        for line in hmmscan_output:
            if line[0] != "#":
                values = line.split()
                pfam_domain.append(values[1])
                seq_id.append(values[2])
                e_value.append(float(values[4]))
    # Creates final output
    # Filters it and writes
    # it to a csv file
    pfam_domains_df = pd.DataFrame(
        {
            "Pfam_domain": pfam_domain,
            "Seq_id": seq_id,
            "E_value": e_value,
        }
    )
    pfam_domains_df["Pfam_domain"] = pfam_domains_df[
        "Pfam_domain"
    ].apply(lambda x: x.split(".")[0])
    pfam_domains_df[pfam_domains_df["E_value"] < 10 ** (-5)][
        ["Pfam_domain", "Seq_id"]
    ].to_csv(path_to_output / f"{filename}.csv", index=None)
    return path_to_output / f"{filename}.csv"
