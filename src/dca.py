# -*- coding: utf-8 -*-
"""This module contains a class to store DCA models.

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

from typing import Tuple, List, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class Dca:
    """Store and manipulate a DCA model."""

    def __init__(
        self, h_matrix: np.ndarray = None, j_matrix: np.ndarray = None
    ):
        """
        Initialize a new Dca instance.

        Parameters
        ----------
        h_matrix : numpy.ndarray, optional
            h matrix of DCA model
        j_matrix : numpy.ndarray, optional
            J matrix of DCA model
        """
        self.h_matrix = h_matrix
        self.j_matrix = j_matrix
        if not self.has_valid_types:
            raise TypeError(
                "Arguments have no valid type for\
                Dca instance attributes."
            )

    def has_valid_types(self) -> bool:
        """
        Check that arguments have valid type to become Dca attributes.

        Returns
        -------
        bool
            Test variable types are valid
        """
        if not isinstance(self.h_matrix, np.ndarray) and (
            self.h_matrix is not None
        ):
            return False
        if not isinstance(self.j_matrix, np.ndarray) and (
            self.j_matrix is not None
        ):
            return False
        return True

    def shape(self) -> Tuple[int, int]:
        """Return instance dimensions.

        Retrieve h and J matrices
        dimensions and check that
        they are compatible with a
        DCA model format.

        Returns
        -------
        Tuple[int]
            DCA model dimensions.
        """
        if self.h_matrix is not None and self.j_matrix is not None:
            h_aa = self.h_matrix.shape[0]
            j_aa = self.j_matrix.shape[0]
            if j_aa != self.j_matrix.shape[1]:
                raise ValueError("J matrix has wrong dimensions.")
            if j_aa != h_aa:
                raise ValueError(
                    "h and J matrices have incompatible dimensions."
                )
            h_seq = self.h_matrix.shape[1]
            j_seq = self.j_matrix.shape[2]
            if j_seq != self.j_matrix.shape[3]:
                raise ValueError("J matrix has wrong dimensions.")
            if j_seq != h_seq:
                raise ValueError(
                    "h and J matrices have incompatible dimensions."
                )
            return h_aa, h_seq
        return 0, 0

    def is_complete(self) -> bool:
        """Check that Dca attributes match requirements.

        Returns
        -------
        bool
            Test result
        """
        if not self.has_valid_types():
            return False
        if self.h_matrix is None:
            return False
        if self.j_matrix is None:
            return False
        if (
            not self.h_matrix.shape[0]
            == self.j_matrix.shape[0]
            == self.j_matrix.shape[1]
        ):
            return False
        if (
            not self.h_matrix.shape[1]
            == self.j_matrix.shape[2]
            == self.j_matrix.shape[3]
        ):
            return False
        return True

    def get_score(
        self, sequence: str, amino_acids: List[str]
    ) -> Union[float, None]:
        """
        Compute sequence DCA score.

        Return sequence DCA score or None
        if sequence has any invalid amino acid.

        Parameters
        ----------
        sequence : str
            Input sequence
        AAs: List[str]
            list of amino acids in the
            same order than in h and J
            entries

        Returns
        -------
        float
            Sequence DCA score.
        """
        if self.h_matrix is not None and self.j_matrix is not None:
            energy = 0
            for index_1, amino_acid_1 in enumerate(sequence):
                energy -= self.h_matrix[
                    amino_acids.index(amino_acid_1), index_1
                ]
                for index_2, amino_acid_2 in enumerate(sequence):
                    if index_2 > index_1:
                        energy -= self.j_matrix[
                            amino_acids.index(amino_acid_1),
                            amino_acids.index(amino_acid_2),
                            index_1,
                            index_2,
                        ]
            return energy
        return None

    def get_single_mutant_scores(
        self, sequence: str, amino_acids: List[str]
    ) -> pd.DataFrame:
        """
        Compute DCA scores of single mutants.

        Return a dataframe of DCA scores of
        all single mutants of input sequence
        or None if sequence has any invalid
        amino acids.

        Parameters
        ----------
        sequence : str
            Input sequence
        AAs: List[str]
            list of amino acids in the
            same order than in h and J
            entries

        Returns
        -------
        pd.DataFrame
            Dataframe of DCA scores for
            all single mutants.
        """
        if self.h_matrix is not None and self.j_matrix is not None:
            # Convert amino acids char to
            # their index values in AAs list.
            seq_int = list(map(amino_acids.index, sequence))
            # Starts single mutant
            # score computation
            score_matrix = np.zeros((len(seq_int), len(amino_acids)))
            # Iterates over positions and
            # values in the sequence
            # to generate mutations
            for mut_index, native_aa_1 in enumerate(seq_int):
                # Iterates over all possible amino
                # acids to generate mutations
                for mut_aa in range(len(amino_acids)):
                    # Computes single mutant score
                    # for mut_AA at mut_index
                    energy = (
                        self.h_matrix[native_aa_1, mut_index]
                        - self.h_matrix[mut_aa, mut_index]
                    )
                    for index in range(mut_index):
                        native_aa_2 = seq_int[index]
                        energy += (
                            self.j_matrix[
                                native_aa_2,
                                native_aa_1,
                                index,
                                mut_index,
                            ]
                            - self.j_matrix[
                                native_aa_2, mut_aa, index, mut_index
                            ]
                        )
                    for index in range(mut_index + 1, len(seq_int)):
                        native_aa_2 = seq_int[index]
                        energy += (
                            self.j_matrix[
                                native_aa_1,
                                native_aa_2,
                                mut_index,
                                index,
                            ]
                            - self.j_matrix[
                                mut_aa, native_aa_2, mut_index, index
                            ]
                        )
                    # Stores the score in the matrix
                    score_matrix[mut_index, mut_aa] = energy
            # Turns the matrix into a dataframe
            dca_scores = pd.DataFrame(score_matrix)
            dca_scores.columns = amino_acids
            return dca_scores
        return None
