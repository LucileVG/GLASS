# -*- coding: utf-8 -*-
"""This module allows to simulate mutations according to neutral models of molecular evolution.

Example
-------
TODO

Notes
-----
TODO

Attributes
----------
TODO
"""

from typing import List

import random
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from src.config import random_seed, translation_table, AAs  # type: ignore   # pylint: disable=import-error
from src.bio_basics import translate  # type: ignore   # pylint: disable=import-error


class NucleotideSubstitution:
    """
    Defines a Markov model of nucleotide substitution.

    Parameters
    ----------
    probability_matrix: pandas.DataFrame
        Dataframe with nucleotide substitution
        probabilities. Indexes and columns
        should be "A", "C", "G", "T". Each row
        should sum to 1.

    Attributes
    ----------
    probability_matrix: pandas.DataFrame
        Dataframe with nucleotide substitution
        probabilities. Indexes and columns
        should be "A", "C", "G", "T". Each row
        should sum to 1.

    Methods
    -------
    generate_mutations
    TODO

    Raises
    ------
    TypeError
        when probability_matrix is not of type pandas.DataFrame
    ValueError
        when probability_matrix.shape is not
        (4,4), or indexes or columns are not
        ["A", "C", "G", "T"] or rows do not
        sum to 1.
    """

    AAs = AAs
    Translation_table = translation_table
    seed = random_seed
    random.seed(a=seed)

    def __init__(self, probability_matrix: pd.DataFrame) -> None:
        """
        Initialize a new NucleotideSubstitution instance.

        Parameters
        ----------
        probability_matrix : pd.DataFrame
            Probability matrix of
            nucleotide substitutions

        Raises
        ------
        TypeError
            If probability_matrix not of
            pd.DataFrame type
        ValueError
            If probability_matrix has not proper
            shape
        ValueError
            If probability_matrix indices are
            valid nucleotides
        ValueError
            If probability_matrix columns are
            valid nucleotides
        ValueError
            If probability_matrix rows are not
            normalized
        """
        if not isinstance(probability_matrix, pd.DataFrame):
            raise TypeError(
                "probability_matrix should be of type pandas.DataFrame"
            )
        if probability_matrix.shape != (4, 4):
            raise ValueError(
                "probability_matrix should be of size (4,4)"
            )
        if set(probability_matrix.index) != set(["A", "C", "G", "T"]):
            raise ValueError(
                "probability_matrix indices should be 'A', 'C', 'G', 'T'"
            )
        if set(probability_matrix.columns) != set(["A", "C", "G", "T"]):
            raise ValueError(
                "probability_matrix columns should be 'A', 'C', 'G', 'T'"
            )
        epsilon = 0.00001
        if not (
            np.abs(probability_matrix.sum(axis=1) - 1) < epsilon
        ).all():
            raise ValueError("probability_matrix rows should sum to 1")
        self.probability_matrix = probability_matrix

    def generate_mutations(
        self, sequence: str, n_mut: int, loci: List[int]
    ) -> pd.DataFrame:
        """
        Initialize a new NucleotideSubstitution instance.

        Parameters
        ----------
        sequence : str
            Initial DNA sequence
        n_mut : int
            Number of mutations
            to simulate
        loci : List[int]
            Amino acid loci to
            cover

        Returns
        -------
        pd.DataFrame
            Dataframe of mutations
            ("Locus","AA1","AA2")
        """
        aa_sequence = translate(sequence)
        if aa_sequence == "":
            raise ValueError(
                "sequence is not a valid coding DNA sequence,\
                it cannot be translated into an amino acid sequence"
            )
        if "*" in aa_sequence:
            raise ValueError("sequence should not contain stop codon")
        # Initializes a matrix to
        # store mutations. Initial
        # entries are 1 for possible
        # mutations and 0 for native
        # amino acid.
        mut_df = pd.DataFrame({"Locus": [], "AA1": [], "AA2": []})
        # Starts simulating mutations
        while len(mut_df) < n_mut:
            # Draws a random codon in the sequence
            codon_pos = loci[random.randrange(len(loci))]
            # Extracts corresponding codon
            codon = sequence[3 * (codon_pos - 1) : 3 * codon_pos]
            # Draws a random position on the codon
            nucleotide_pos = random.randrange(3)
            # Extracts the corresponding nucleotide
            nucleotide = codon[nucleotide_pos]
            # Draws a mutated nucleotide
            # with probabilities registered
            # in the probability matrix
            mut_nucleotide = self.probability_matrix.columns[
                list(
                    np.cumsum(
                        self.probability_matrix.loc[nucleotide, :]
                    )
                    > random.random()
                ).index(True)
            ]
            # Computes the corresponding
            # mutated codon
            mut_codon_bases = list(codon)
            mut_codon_bases[nucleotide_pos] = mut_nucleotide
            mut_codon = "".join(mut_codon_bases)
            # Retrieves corresponding
            # amino acid
            mut_amino_acid = self.Translation_table[mut_codon]
            if mut_amino_acid in self.AAs:
                # Registers mutation
                # in the matrix
                mut_df = mut_df.append(
                    {
                        "Locus": codon_pos,
                        "AA1": aa_sequence[codon_pos - 1],
                        "AA2": mut_amino_acid,
                    },
                    ignore_index=True,
                ).drop_duplicates()
                mut_df = mut_df[mut_df["AA1"] != mut_df["AA2"]]
        return mut_df


class JC69(NucleotideSubstitution):
    """
    Defines a Jukes and Cantor 1969 (JC69) model of nucleotide substitution.

    Attributes
    ----------
    probability_matrix: pandas.DataFrame
        Dataframe with equal nucleotide
        substitution probabilities.
    """

    def __init__(self):
        """Initialize a new JC69 instance."""
        probability_matrix = pd.DataFrame(
            (np.ones((4, 4)) - np.eye(4)) * 1 / 3
        )
        probability_matrix.index = ["A", "C", "G", "T"]
        probability_matrix.columns = ["A", "C", "G", "T"]
        NucleotideSubstitution.__init__(self, probability_matrix)


class K80(NucleotideSubstitution):
    """
    Defines a Kimura 1980 (K80) model of nucleotide substitution.

    Parameters
    ----------
    kappa: float
        transition to transversion rate ratio

    Attributes
    ----------
    probability_matrix: pandas.DataFrame
        Dataframe with nucleotide substitution
        probabilities built from transition
        to transversion rate ratio kappa.
    """

    def __init__(self, kappa: float):
        """
        Initialize a new JC69 instance.

        Parameters
        ----------
        kappa : float
            Transition to transversion
            rate ratio

        Raises
        ------
        TypeError
            If kappa is not of type float
        """
        if not isinstance(kappa, float):
            raise TypeError("parameter kappa should be of type float")
        ts_p = kappa / (kappa + 2)
        tv_p = 1 / (kappa + 2)
        probability_matrix = pd.DataFrame(
            np.array(
                [
                    [0, tv_p, ts_p, tv_p],
                    [tv_p, 0, tv_p, ts_p],
                    [ts_p, tv_p, 0, tv_p],
                    [tv_p, ts_p, tv_p, 0],
                ]
            )
        )
        probability_matrix.index = ["A", "C", "G", "T"]
        probability_matrix.columns = ["A", "C", "G", "T"]
        NucleotideSubstitution.__init__(self, probability_matrix)
