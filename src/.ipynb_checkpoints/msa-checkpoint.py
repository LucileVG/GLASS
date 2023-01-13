# -*- coding: utf-8 -*-
"""This module contains a class to store and process MSAs.
"""


from pathlib import Path
from typing import Union, List
import subprocess

from Bio import SeqIO, AlignIO  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from src.config import (  # type: ignore # pylint: disable=import-error
    AAs,
    N_MSA,
    prop_gap,
    priority_coefs,
)


class Msa:
    """
    Stores and manipulates Multiple Sequence Alignments (MSAs).

    Attributes
    ----------
    N_MSA : int
        The minimal number of sequences in a training MSA.
    prop_gap : float
        The maximal proportion of gaps allowed in a MSA
        column, columns with more than prop_gap gaps are
        removed. Should be comprised between 0 and 1.
    valid_AAs : List[str]
        Valid characters for amino acids.
    valid_MSA_origins : List[str]
        Valid str values for MSA_type attribute.

    Methods
    -------
    read_msa
        Reads MSA in a fasta file and returns it as a numpy
        array if it contains at least N_MSA sequences.
    remove_bad_aa
        Changes unvalid amino acids to
        gaps in MSA.
    TODO
    """

    N_MSA: int = N_MSA
    prop_gap: float = prop_gap
    valid_AAs: List[str] = AAs
    valid_MSA_origins: List[str] = list(priority_coefs.keys())

    def __init__(self, array: np.ndarray = None, origin: str = None):
        """
        Initialize a new Msa instance.

        Parameters
        ----------
        array : numpy.ndarray, optional
            MSA of protein distant homologs
        origin : str, optional
            Origin of the MSA (should belong to the list of valid_MSA_origins)
        """
        self.array = array
        self.origin = origin
        if not self.has_valid_types():
            raise TypeError(
                "Arguments have no valid type for\
                Msa instance attributes."
            )

    def has_valid_types(self) -> bool:
        """Validate types.

        Check that arguments have valid
        type to become Msa attributes.
        If accept_none is True, None
        types are considered as valid
        attribute types.

        Returns
        -------
        bool
            Test variable types are valid
        """
        if not isinstance(self.array, np.ndarray) and (
            self.array is not None
        ):
            return False
        if not isinstance(self.origin, str) and (
            self.origin is not None
        ):
            return False
        return True

    def read_msa(
        self, path: Union[str, Path]
    ) -> Union[np.ndarray, None]:
        """Read and return MSA.

        Read MSA in the fasta file located
        at the input path. Return a numpy
        array containing the corresponding
        MSA if there are at least N_MSA
        sequences in the MSA, None otherwise.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the input fasta file

        Returns
        -------
        numpy.ndarray
            Array containing the MSA

        Raises
        ------
        ValueError
            when MSA sequences have different lengths
        """
        # Read MSA in fasta file and generate corresponding np.array
        with open(path, "r") as input_file:
            sequences = [
                list(str(record.seq).upper())
                for record in SeqIO.parse(input_file, "fasta")
            ]
            # Number of sequences in MSA should be above N_MSA threshold
            if len(sequences) < self.N_MSA:
                return None
            length = len(sequences[0])
            # Check all sequences in MSA have same length
            if not all(len(seq) == length for seq in sequences):
                raise ValueError(
                    "not all sequences\
                    have same length in the MSA"
                )
            return np.array(sequences)

    def retrieve_pfam_msa(self, pfam_id: str, path: Path) -> None:
        """Retrieve a MSA from PFAM.

        Download PFAM MSA and HMM if
        these are not already downloaded.
        Strip inserts from PFAM MSA and
        loads it as instance attribute
        self.array.

        Parameters
        ----------
        pfam_id : str
            PFAM id
        path : pathlib.Path
            Path to the folder where
            downloaded files are stored

        Returns
        -------
        NoneType
            None
        """
        # Download PFAM MSA
        # if not already done
        success = True
        if not pfam_msa_downloaded(pfam_id, path):
            success = download_pfam_msa(pfam_id, path)
        if success:
            # Read PFAM MSA
            pfam_msa = np.array(
                [
                    list(str(seq.seq))
                    for seq in AlignIO.read(
                        path / f"{pfam_id}_full.sth", "stockholm"
                    )
                ]
            )
            # Strip columns that correspond to inserts
            keep_col = list()
            for j in range(pfam_msa.shape[1]):
                col = "".join(pfam_msa[:, j])
                if col == col.upper():
                    keep_col.append(j)
            # Update MSA attribute
            self.array = pfam_msa[:, keep_col]
        else:
            self.array = None

    def remove_bad_aa(self) -> None:
        """Remove unknown amino acids.

        Modify self.array by changing
        all non-valid amino acid
        characters by gaps. Be aware
        that the method is case-sensitive.

        Returns
        -------
        NoneType
            None
        """
        clean_aa = np.vectorize(
            lambda x: x if x in self.valid_AAs else "-"
        )
        self.array = clean_aa(self.array)

    def is_complete(self) -> bool:
        """Check that Msa attributes match requirements.

        Returns
        -------
        bool
            Test result
        """
        # Check attribute types
        if not self.has_valid_types():
            return False
        # Check that there are enough
        # sequences in MSA
        if (
            not isinstance(self.array, np.ndarray)
            or self.array.shape[0] < self.N_MSA
        ):
            return False
        # Check that all amino acids in MSA are
        # valid amino acids
        if not np.all(np.isin(self.array, self.valid_AAs)):
            return False
        # Check that too gapped columns are removed
        # from MSA
        if not np.all(
            self.array.shape[0] - (self.array == "-").sum(axis=0)
            >= (1 - self.prop_gap) * self.N_MSA
        ):
            return False
        # Check that MSA origin is a valid origin
        if self.origin not in self.valid_MSA_origins:
            return False
        return True

    def to_fasta(self, path: Union[str, Path]) -> None:
        """
        Write MSA array to fasta file at input path.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the fasta file
            where MSA should be written

        Returns
        -------
        NoneType
            None
        """
        if self.array is not None:
            with open(path, "w") as output_file:
                for i in range(self.array.shape[0]):
                    output_file.write(
                        f">seq_{i}\n{''.join(self.array[i,:])}\n"
                    )

    def get_ind_model(
        self, amino_acids: List[str]
    ) -> Union[pd.DataFrame, None]:
        """
        Compute IND model.

        Compute IND model from
        array attribute. Returns
        None if array is None.

        Parameters
        ----------
        amino_acids: List[str]
            list of amino acids

        Returns
        -------
        np.ndarray
            IND model
        """
        frequencies = list()
        msa_int = np.vectorize(amino_acids.index)(self.array)
        for column in range(msa_int.shape[1]):
            values, counts = np.unique(
                msa_int[:, column], return_counts=True
            )
            aa_freq = np.ones(len(amino_acids))
            for index, value in enumerate(values):
                aa_freq[value] += counts[index]
            aa_freq /= aa_freq.sum()
            frequencies.append(aa_freq)
        return pd.DataFrame(
            -np.log10(np.array(frequencies)), columns=amino_acids
        )

    def get_score(
        self, sequence: str, amino_acids: List[str]
    ) -> Union[float, None]:
        """
        Compute sequence IND score.

        Return sequence IND score or None
        if sequence has any invalid amino acid.

        Parameters
        ----------
        sequence : str
            Input sequence
        amino_acids: List[str]
            list of amino acids

        Returns
        -------
        float
            Sequence IND score.
        """
        # Compute independent model
        ind_model = self.get_ind_model(amino_acids)
        if ind_model is not None:
            # Compute sequence matrix
            seq_int = list(map(amino_acids.index, sequence))
            sequence_matrix = np.zeros((len(seq_int), len(amino_acids)))
            for index, amino_acid in enumerate(seq_int):
                sequence_matrix[index, amino_acid] = 1
            # Compute sequence score
            return (sequence_matrix * ind_model).sum()
        return None

    def get_single_mutant_scores(
        self, sequence: str, amino_acids: List[str]
    ) -> pd.DataFrame:
        """
        Compute IND scores of single mutants.

        Return a dataframe of IND scores of
        all single mutants of input sequence
        or None if sequence has any invalid
        amino acids.

        Parameters
        ----------
        sequence : str
            Input sequence
        AAs: List[str]
            list of amino acids

        Returns
        -------
        pd.DataFrame
            Dataframe of IND scores for
            all single mutants.
        """
        # Compute independent model
        ind_model = self.get_ind_model(amino_acids)
        if ind_model is not None:
            # Starts single mutant
            # score computation
            score_matrix = np.zeros((len(sequence), len(amino_acids)))
            # Iterates over positions and
            # values in the sequence
            # to generate mutations
            for mut_index, native_aa in enumerate(sequence):
                # Iterates over all possible amino
                # acids to generate mutations
                for aa_index, mut_aa in enumerate(amino_acids):
                    # Computes single mutant score
                    # for mut_AA at mut_index
                    score_matrix[mut_index, aa_index] = (
                        ind_model.loc[mut_index, mut_aa]
                        - ind_model.loc[mut_index, native_aa]
                    )
            return pd.DataFrame(score_matrix, columns=amino_acids)
        return None


def pfam_msa_downloaded(pfam_id: str, path: Path) -> bool:
    """Check existence of PFAM MSA.

    Check whether MSA corresponding
    to input PFAM id is already present
    at input path.

    Parameters
    ----------
    pfam_id : str
        PFAM id
    path : pathlib.Path
        Path to the folder where
        downloaded files are stored

    Returns
    -------
    bool
        True if files are present,
        False otherwise
    """
    return (path / f"{pfam_id}_full.sth").is_file()


def pfam_hmm_downloaded(pfam_id: str, path: Path) -> bool:
    """Check existence of PFAM HMM.

    Check whether HMM corresponding
    to input PFAM id is already present
    at input path.

    Parameters
    ----------
    pfam_id : str
        PFAM id
    path : pathlib.Path
        Path to the folder where
        downloaded files are stored

    Returns
    -------
    bool
        True if files are present,
        False otherwise
    """
    return (path / f"hmm_{pfam_id}").is_file()


def download_pfam_msa(pfam_id: str, path: Path) -> bool:
    """Download PFAM MSA.

    Download PFAM full MSA for a given
    PFAM id. File is saved in folder
    at input path.

    Parameters
    ----------
    pfam_id : str
        PFAM id
    path : pathlib.Path
        Path to the folder where
        downloaded files are stored

    Returns
    -------
    bool
        True if success, False otherwise
    """
    # Download PFAM MSA
    path_pfam_msa = (
        f"http://pfam.xfam.org/family/{pfam_id}/alignment/full/gzipped"
    )
    try:
        subprocess.run(
            [
                "curl",
                "-o",
                path / f"{pfam_id}_full.sth.gz",
                path_pfam_msa,
            ],
            check=True,
        )
        # Gunzip PFAM MSA
        if (path / f"{pfam_id}_full.sth.gz").is_file():
            subprocess.run(
                ["gunzip", path / f"{pfam_id}_full.sth.gz"], check=True
            )
            return True
    except:  # pylint: disable=bare-except
        return False
    return False
