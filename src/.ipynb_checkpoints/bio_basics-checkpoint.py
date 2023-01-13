# -*- coding: utf-8 -*-
"""Implement generalist functions to deal with biological sequences."""

from pathlib import Path
from typing import List, Dict, Union
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore
from src.config import translation_table, codons, AAs  # type: ignore   # pylint: disable=import-error


def read_fasta(path: Union[str, Path]) -> List:
    """Read input fasta file.

    Read sequences contained in
    input fasta file and returns
    them in a list.

    Parameters
    ----------
    path : str, pathlib.Path
        Path to the input fasta file

    Returns
    -------
    List
        List of sequences contained
        in the fasta file
    """
    with open(path, "r") as input_file:
        # Parse file with Biopython
        # SeqIO parser.
        return list(SeqIO.parse(input_file, "fasta"))


def write_fasta(sequences: List, path: Union[str, Path]) -> None:
    """Write sequences to fasta file.

    Parameters
    ----------
    sequences : List
        List of sequences as SeqRecord
        objects
    path : str, pathlib.Path
        Path to the input fasta file

    Returns
    -------
    NoneType
        None
    """
    with open(path, "w") as output_file:
        SeqIO.write(sequences, output_file, "fasta")


def read_fasta_dict(path: Union[str, Path]) -> Dict:
    """Read input fasta file.

    Read sequences contained in
    input fasta file and returns
    them in a dict {id: sequence record}.

    Parameters
    ----------
    path : str, pathlib.Path
        Path to the input fasta file

    Returns
    -------
    Dict
        Dict of sequences contained
        in the fasta file indexed by
        sequence ids
    """
    with open(path, "r") as input_file:
        # Parse file with Biopython
        # SeqIO parser.
        return {
            record.id: record
            for record in SeqIO.parse(input_file, "fasta")
        }


def translate(sequence: str) -> str:
    """Translate sequence.

    Check that the input sequence
    is a valid gene sequence, i.e.
    all positions are valid codons.
    If so, translates it into and
    amino-acid sequence. Otherwise
    returns an empty string.

    Parameters
    ----------
    sequence : str
        DNA sequence

    Returns
    -------
    str
        Amino acid sequence
    """
    # Check that sequence length
    # is a multiple of 3.
    if len(sequence) % 3 != 0:
        return ""
    # Initialize an empty AA sequence
    aa_sequence = ""
    # Iterate over sequence codons
    for i in range(0, len(sequence) - 1, 3):
        codon = sequence[i : i + 3]
        # Check the codon is valid
        if codon in translation_table.keys():
            aa_sequence += translation_table[codon]
        else:
            aa_sequence += "X"
    return aa_sequence


def reverse_aln(dna_seq: str, aln_aa_seq: str) -> str:
    """Reverse-align a DNA sequence from an aligned AA sequence.

    Parameters
    ----------
    dna_seq : str
        Raw DNA sequence
    aln_aa_seq : str
        Aligned AA sequence

    Returns
    -------
    str
        Aligned DNA sequence
    """
    start_gap = len(aln_aa_seq) - len(
        aln_aa_seq.lstrip("-")
    )  # Number of gaps at the beginning of the aligned protein sequence
    end_gap = len(aln_aa_seq) - len(
        aln_aa_seq.rstrip("-")
    )  # Number of gaps at the end of the aligned protein sequence
    aln_aa_seq = aln_aa_seq.lstrip("-").rstrip(
        "-"
    )  # We remove starting and ending gaps
    values = [len(AA) * 3 for AA in aln_aa_seq.split("-")]
    # Each element of the list is the number of aminoacids
    # between two consecutive gaps (can be 0 if there are
    # several gaps one after the other) multiplied by 3
    # in order to have the number of nucleotides
    aln_dna_seq = (
        "---" * start_gap
    )  # Start the aligned nucleotidic sequence
    count = 0
    for bps in values:  # Loop to add nucleotides and gaps
        aln_dna_seq = aln_dna_seq + dna_seq[count : count + bps] + "---"
        count += bps
    return (
        aln_dna_seq.rstrip("---") + "---" * end_gap
    )  # Add the ending gaps


def generate_codons_df(sequence: str) -> pd.DataFrame:
    """Transform input sequence into a codon matrix.

    Parameters
    ----------
    sequence : str
        Input DNA sequence

    Returns
    -------
    pandas.DataFrame
        Codon matrix
    """
    codons_dict: Dict[str, List] = {"Locus": [], "Codon": []}
    for i in range(0, len(sequence), 3):
        codon = sequence[i : i + 3].upper()
        if codon in codons:
            codons_dict["Locus"].append(i // 3 + 1)
            codons_dict["Codon"].append(codon)
    return pd.DataFrame(codons_dict)


def pivot_codons_df(codons_df: pd.DataFrame) -> pd.DataFrame:
    """Transform codon matrix into codon dataframe.

    Parameters
    ----------
    codons_df : pandas.DataFrame
        Input codon matrix

    Returns
    -------
    pandas.DataFrame
        Codon dataframe
    """
    codons_df["Value"] = 1
    codons_df = codons_df.pivot(
        index="Locus", columns="Codon", values="Value"
    ).fillna(0)
    for codon in codons:
        if codon not in codons_df.columns:
            codons_df[codon] = 0
    return codons_df[codons]


def codon_to_aa_df(codons_df: pd.DataFrame) -> pd.DataFrame:
    """Translate a codon matrix into an amino acid matrix.

    Parameters
    ----------
    codons_df : pandas.DataFrame
        Input codon matrix

    Returns
    -------
    pandas.DataFrame
        Corresponding amino acid
        matrix
    """
    real_amino_acids = [aa for aa in AAs if aa != "-"]
    aa_df = pd.DataFrame(
        np.zeros((len(codons_df), len(real_amino_acids))),
        columns=real_amino_acids,
        index=codons_df.index,
    )
    for codon in codons_df.columns:
        amino_acid = translate(codon)
        if amino_acid in real_amino_acids:
            aa_df[amino_acid] += codons_df[codon]
    return aa_df
