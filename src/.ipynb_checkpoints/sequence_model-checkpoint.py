# -*- coding: utf-8 -*-
"""This module contains a class to initialize, train and store amino acid sequence models.

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

from pathlib import Path
from typing import List, Optional, Dict, Union
import subprocess

from Bio.SeqRecord import SeqRecord  # type: ignore
from Bio.Seq import Seq  # type: ignore
from Bio import AlignIO  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from src.config import (  # type: ignore # pylint: disable=import-error
    AAs,
    prop_id,
    hmmalign,
    julia,
    plmdca,
    hhblits,
    uniclust,
)
from src.bio_basics import read_fasta, write_fasta  # type: ignore # pylint: disable=import-error
from src.dca import Dca  # type: ignore # pylint: disable=import-error
from src.msa import Msa  # type: ignore # pylint: disable=import-error


class SequenceModel:
    """Store and manipulate a protein model.

    Gather a protein model (DCA or other)
    with its corresponding training MSA
    and reference sequence. Provide
    methods to perform quality checks,
    read and write models.

    Attributes
    ----------
    valid_AAs : List[str]
        Valid characters for amino acids.
    prop_id : float
        The identity threshold above which sequences that
        are too close from reference sequence are removed
        from training MSA. Should be comprised between 0
        and 1.

    Methods
    -------
    extract_aln_seq
        Extracts aligned part of the reference sequence.
    is_similar
        Computes sequence identity with reference sequence.
    is_complete
        Checks that DcaModel instance meets all requirements.
    write
        Writes DcaModel instance to output folder if it meets
        all requirements.
    gather_msa
        Gathers sequences from distant
        homologs of the reference sequence
        to form a MSA.
    train_dca
        Trains DCA model from MSA.
    filter_msa
        Filters MSA to match requirements
    TODO
    """

    valid_AAs: List[str] = AAs
    prop_id: float = prop_id

    def __init__(
        self,
        name: str,
        ref_sequence: str = None,
        model: Dca = None,
        msa: Msa = None,
    ) -> None:
        """
        Initialize a new SequenceModel instance.

        Parameters
        ----------
        name : str, optional
            name of the model
        ref_sequence : str, optional
            reference amino acid sequence with
            aligned parts in upper char and unaligned
            parts in lower char
        model : Dca, optional
            corresponding DCA model
        msa : Msa, optional
            MSA aligning with aligned part of ref_sequence
        """
        self.name = name
        self.ref_sequence = ref_sequence
        self.model = model
        self.msa = msa
        self._scores: Dict[str, pd.DataFrame] = dict()
        if not self.has_valid_types():
            raise TypeError(
                "Arguments have no valid type for\
                SequenceModel instance attributes."
            )

    def has_valid_types(self) -> bool:
        """
        Check that arguments have valid type to become SequenceModel attributes.

        Returns
        -------
        bool
            Test variable types are valid
        """
        if not isinstance(self.name, str) and (self.name is not None):
            return False
        if not isinstance(self.ref_sequence, str) and (
            self.ref_sequence is not None
        ):
            return False
        if not isinstance(self.model, Dca) and (self.model is not None):
            return False
        if not isinstance(self.msa, Msa) and (self.model is not None):
            return False
        return True

    def extract_aln_seq(self) -> str:
        """Extract aligned part of an amino acid sequence.

        Extract the subsequence corresponding to the
        loci aligned to a given domain (upper case
        characters only, gaps removed) in the reference
        sequence. Return a string corresponding to
        the subsequence.

        Returns
        -------
        str
            Aligned subsequence
        """
        if isinstance(self.ref_sequence, str):
            return "".join(
                [i for i in self.ref_sequence if i.upper() == i]
            ).replace("-", "")
        raise TypeError(
            "ref_sequence attribute should be a string not None"
        )

    def is_similar(self, sequence: str) -> bool:
        """Check identity between sequence and self.ref_sequence.

        Check that sequence has same length
        than aligned part of reference sequence
        and computes sequence identity between
        both. Return True if sequence identity
        >= prop_id and False otherwise.

        Parameters
        ----------
        sequence : str
            Input sequence

        Returns
        -------
        bool
            Sequence identity >= prop_id

        Raises
        ------
        ValueError
            sequence length does not match length of
            the aligned part of reference sequence
        """
        # Extract the subsequence of the reference
        # sequence which is aligned to the model
        aln_ref_sequence = self.extract_aln_seq()
        # Compute sequence length
        length = len(sequence)
        # Check that lengths are equal
        if length != len(aln_ref_sequence):
            raise ValueError(
                "sequence length does\
                not match reference sequence length"
            )
        #  Compute sequence identity
        identity_seq = (
            sum(
                (
                    int(sequence[i] == aln_ref_sequence[i])
                    for i in range(length)
                )
            )
            / length
        )
        # Return True if sequence identity >= prop_id
        return identity_seq >= self.prop_id

    def is_complete(self) -> bool:
        """Check that SequenceModel attributes match requirements.

        Check that SequenceModel attributes
        match requirements. Sizes should
        match the length of the aligned part
        of the reference sequence. The aligned
        part of the reference sequence should
        only contain valid amino acids.

        Returns
        -------
        bool
            Test result
        """
        # Check that attributes have correct type
        # Should not be None
        if (
            not self.has_valid_types()
            or self.msa is None
            or self.model is None
        ):
            return False
        # Check that MSA and DCA are complete
        if not self.msa.is_complete() or not self.model.is_complete():
            return False
        # Extract the subsequence of the reference
        # sequence which is aligned to the model
        aln_ref_sequence = self.extract_aln_seq()
        # Check the subsequence only has valid
        # amino acids
        if not is_valid_seq(aln_ref_sequence, self.valid_AAs):
            return False
        # Check that all sizes match requirements
        length = len(aln_ref_sequence)
        if self.model.shape() != (len(self.valid_AAs), length,):
            return False
        # Check that MSA sequences have
        # length match aligned part of
        # reference sequence length
        # Check that sequences too close to reference
        # sequence are removed from MSA
        if (
            not isinstance(self.msa.array, np.ndarray)
            or self.msa.array.shape[1] != length
            or not np.all(
                ~np.apply_along_axis(
                    self.is_similar, -1, self.msa.array
                )
            )
        ):
            return False
        return True

    def write(self, output_folder: Path):
        """Write SequenceModel instance to a .npz file.

        Check that SequenceModel instance matches
        requirements. If so, writes it to a .npz
        file using the model name as file name.

        Parameters
        ----------
        output_folder: Path
            Path to output folder

        Returns
        -------
        NoneType
            None
        """
        if (
            self.is_complete()
            and self.model is not None
            and self.msa is not None
        ):
            np.savez(
                output_folder / f"{self.name}.npz",
                reference_sequence=self.ref_sequence,
                h=self.model.h_matrix,
                j=self.model.j_matrix,
                msa_array=self.msa.array,
                msa_type=self.msa.origin,
            )
        else:
            with open(
                output_folder / f"{self.name}.txt", "w"
            ) as output_file:
                if self.msa is not None and self.msa.array is not None:
                    output_file.write(f"{self.msa.array.shape}")
                else:
                    output_file.write("No MSA")

    def run_hhblits(self, path: Path) -> None:
        """Run HHblits to find distant homologs MSA.

        Parameters
        ----------
        path : Path
            Path to temporary folder

        Returns
        -------
        NoneType
            None
        """
        self.msa = Msa()
        # Write reference sequence to
        # fasta file to run hhblits
        path_refseq = path / f"{self.name}.fasta"
        path_hhr = path / f"{self.name}.hhr"
        path_aln = path / f"{self.name}.a3m"
        with open(path_refseq, "w") as refseq_file:
            refseq_file.write(
                f">Reference_sequence\n{self.ref_sequence}"
            )
        subprocess.run(
            [
                hhblits,
                "-i",
                path_refseq,
                "-o",
                path_hhr,
                "-oa3m",
                path_aln,
                "-d",
                uniclust,
            ],
            check=True,
        )
        sequences = [
            SeqRecord(
                Seq(
                    "".join([i for i in str(seq.seq) if i.upper() == i])
                ),
                id=seq.id,
                description="",
            )
            for seq in read_fasta(path_aln)
        ]
        write_fasta(sequences, path_aln)
        self.msa.array = self.msa.read_msa(path_aln)
        # Remove useless temporary files
        path_refseq.unlink()
        path_hhr.unlink()
        path_aln.unlink()

    def align_to_pfam_hmm(self, pfam_id: str, path: Path) -> None:
        """Align reference sequence to PFAM HMM.

        Align reference sequence to PFAM HMM.
        Strip gapped positions in the reference
        from PFAM MSA. Update instance attributes
        MSA_array and ref_sequence accordingly.

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
        # Checks whether PFAM hmm exists
        # If not, download it
        success = True
        if not pfam_hmm_downloaded(pfam_id, path):
            success = download_pfam_hmm(pfam_id, path)
        if (
            success
            and self.msa is not None
            and self.msa.array is not None
        ):
            # Write reference sequence to
            # fasta file to run hmmalign
            path_refseq = path / f"{self.name}.fasta"
            path_aln = path / f"{self.name}.sth"
            with open(path_refseq, "w") as refseq_file:
                refseq_file.write(
                    f">Reference_sequence\n{self.ref_sequence}"
                )
            try:
                # Run hmmalign
                with open(path_aln, "w") as output_file:
                    subprocess.run(
                        [
                            hmmalign,
                            path / f"hmm_{pfam_id}",
                            path_refseq,
                        ],
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                # Read hmmalign output
                aln_aa_seq = str(
                    AlignIO.read(path_aln, "stockholm")[0].seq
                )
                # Remove gaps
                not_gapped = [
                    i != "-" for i in aln_aa_seq if i.upper() == i
                ]
                # Checks that length of aligned
                # reference sequence match msa
                if len(not_gapped) == self.msa.array.shape[1]:
                    # Update MSA array attribute
                    self.msa.array = self.msa.array[:, not_gapped]
                # Otherwise remove msa array
                else:
                    self.msa.array = None
                # Update reference sequence attribute
                self.ref_sequence = aln_aa_seq.replace("-", "")
                # Remove useless temporary file
                path_aln.unlink()
            except:  # pylint: disable=bare-except
                self.msa = Msa()
            # Remove useless temporary files
            path_refseq.unlink()
        else:
            self.msa = Msa()

    def filter_msa(self) -> None:
        """Filter MSA for gaps and similarity with reference sequence.

        Filter msa.array attribute to remove
        sequences too close to reference
        sequence and columns that are too
        gapped in the MSA.

        Returns
        -------
        NoneType
            None
        """
        # Compare MSA sequences to
        # reference
        if self.ref_sequence is None:
            raise ValueError(
                "ref_sequence attribute should not be None"
            )
        aln_ref_sequence = "".join(
            [
                amino_acid
                for amino_acid in self.ref_sequence
                if amino_acid.upper() == amino_acid
            ]
        )
        if self.msa is None:
            raise ValueError("msa attribute should not be None")
        if self.msa.array is None:
            raise ValueError("msa.array attribute should not be None")
        if len(aln_ref_sequence) > 0 and self.msa.array.shape[1] == len(
            aln_ref_sequence
        ):
            identity_with_ref = np.apply_along_axis(
                identity, -1, self.msa.array, aln_ref_sequence
            )
            # Remove from MSA, sequences
            # that have >= prop_id identity
            # with reference sequence
            self.msa.array = self.msa.array[
                identity_with_ref < self.prop_id, :
            ]
            # Compute proportion of gaps
            # for each MSA column
            not_gapped_col = self.msa.array.shape[0] - (
                self.msa.array == "-"
            ).sum(
                axis=0
            )  # / self.msa.array.shape[0]
            # Remove from MSA and reference
            # sequence columns that
            # have >= prop_gap gaps
            # self.msa.array = self.msa.array[:, gapped_col < self.msa.prop_gap]
            self.msa.array = self.msa.array[
                :,
                not_gapped_col
                >= (1 - self.msa.prop_gap) * self.msa.N_MSA,
            ]
            new_ref_sequence = ""
            i = 0
            for amino_acid in self.ref_sequence:
                if amino_acid.lower() == amino_acid:
                    new_ref_sequence += amino_acid
                elif (
                    not_gapped_col[i]
                    >= (1 - self.msa.prop_gap) * self.msa.N_MSA
                ):
                    new_ref_sequence += amino_acid
                    i += 1
                else:
                    new_ref_sequence += amino_acid.lower()
                    i += 1
            self.ref_sequence = new_ref_sequence
            # Remove from MSA, sequences
            # that have >= prop_id identity
            # with new reference sequence
            aln_ref_sequence = "".join(
                [
                    amino_acid
                    for amino_acid in self.ref_sequence
                    if amino_acid.upper() == amino_acid
                ]
            )
            if len(aln_ref_sequence) > 0:
                identity_with_ref = np.apply_along_axis(
                    identity, -1, self.msa.array, aln_ref_sequence
                )
                self.msa.array = self.msa.array[
                    identity_with_ref < self.prop_id, :
                ]
                # Checks that there are still
                # enough sequences in MSA
                if self.msa.array.shape[0] < self.msa.N_MSA:
                    self.msa.array = None
            else:
                self.msa.array = None
        else:
            self.msa.array = None

    def gather_msa(
        self, tmp_folder: Path, pfam_id: Optional[str] = None
    ) -> None:
        """Retrieve and clean MSA.

        Retrieve and clean MSA of distant homologs
        using PFAM MSA if pfam_id is a PFAM
        identifier and hhblits if pfam_id is None.

        Parameters
        ----------
        tmp_folder : pathlib.Path
            Path to the folder where
            temporary files are stored
        pfam_id : str
            PFAM id

        Returns
        -------
        NoneType
            None
        """
        self.msa = Msa()
        if isinstance(pfam_id, str):
            self.msa.retrieve_pfam_msa(pfam_id, tmp_folder)
            self.align_to_pfam_hmm(pfam_id, tmp_folder)
            self.msa.origin = "pfam"
        else:
            self.run_hhblits(tmp_folder)
            self.msa.origin = "hhblits"
        if self.msa is None or self.msa.array is None:
            return None
        self.msa.remove_bad_aa()
        self.filter_msa()
        return None

    def train_dca(self, tmp_folder: Path) -> None:
        """Train a DCA model.

        Train a DCA model from msa.array
        attribute and initialize model
        attribute accordingly.

        Parameters
        ----------
        tmp_folder : str, pathlib.Path
            Path to the folder where
            temporary files are stored

        Returns
        -------
        NoneType
            None
        """
        path_msa = tmp_folder / f"{self.name}.fasta"
        path_dca = tmp_folder / f"{self.name}.npz"
        if self.msa is not None and self.msa.array is not None:
            self.msa.to_fasta(path_msa)
            try:
                subprocess.run(
                    [julia, plmdca, path_msa, path_dca], check=True
                )
                dca_model = np.load(path_dca)
                self.model = Dca(
                    h_matrix=dca_model["h"], j_matrix=dca_model["J"]
                )
                path_dca.unlink()
            except:  # pylint: disable=bare-except
                print("Error")
                self.model = Dca()
            if path_msa.is_file():
                path_msa.unlink()

    def get_dca_scores(self) -> pd.DataFrame:
        """
        Compute DCA scores of single mutants of reference sequence.

        Return a dataframe of DCA scores of
        single mutants of reference sequence.

        Returns
        -------
        pd.DataFrame
            Dataframe of DCA scores for
            all single mutants.
        """
        if "DCA" in self._scores.keys():
            return self._scores["DCA"]
        if self.model is not None:
            self._scores["DCA"] = self.model.get_single_mutant_scores(
                self.extract_aln_seq(), self.valid_AAs
            )
            return self._scores["DCA"]
        return None

    def get_ind_scores(self) -> pd.DataFrame:
        """
        Compute IND scores of single mutants of reference sequence.

        Return a dataframe of IND scores of
        single mutants of reference sequence.

        Returns
        -------
        pd.DataFrame
            Dataframe of IND scores for
            all single mutants.
        """
        if "IND" in self._scores.keys():
            return self._scores["IND"]
        if self.msa is not None:
            self._scores["IND"] = self.msa.get_single_mutant_scores(
                self.extract_aln_seq(), self.valid_AAs
            )
            return self._scores["IND"]
        return None

    def get_coverage(self) -> Union[np.ndarray, None]:
        """
        Compute the coverage of the reference sequence by the model.

        Return an array of same length
        than reference sequence where
        entry=0 if site is not covered
        by the model, 1 if it is covered but
        does not match quality criteria and 2
        if it is covered and matches quality
        criteria.

        Returns
        -------
        numpy.ndarray
            Coverage array
        """
        if isinstance(self.ref_sequence, str) and isinstance(
            self.msa, Msa
        ):
            # Computes reference sequence coverage
            coverage = np.array(
                [int(i.upper() == i) for i in self.ref_sequence]
            )
            # Computes proportion of gaps per MSA column
            # and compares it to the prop_gap_MSA threshold
            is_valid_amino_acid = np.vectorize(
                lambda x: int(x in self.valid_AAs and x != "-")
            )
            gap_quality = (
                is_valid_amino_acid(self.msa.array).mean(axis=0)
                * self.msa.array.shape[0]
                >= (1 - self.msa.prop_gap) * self.msa.N_MSA
            ).astype(int) + 1
            # Updates coverage to remove too gapped indices
            j = 0
            for index, value in enumerate(coverage):
                if value == 1:
                    coverage[index] = gap_quality[j]
                    j += 1
            return coverage
        return None


def is_valid_seq(
    sequence: str, alphabet: List[str], case_sensitive: bool = False
) -> bool:
    """Check sequence has a valid characters.

    Check that each position of the
    input sequence is a character.
    Return True if all characters
    are in input alphabet

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    alphabet : List[str]
        List of valid characters
    case_sensitive : bool, optional
        Should the test be case-sensitive (default False)

    Returns
    -------
    bool
        Test result
    """
    # Check all sequence letters are in alphabet
    if case_sensitive:
        for character in sequence:
            if character not in alphabet:
                return False
    upper_alphabet = [character.upper() for character in alphabet]
    for character in sequence.upper():
        if character not in upper_alphabet:
            return False
    return True


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


def download_pfam_hmm(pfam_id: str, path: Path) -> bool:
    """Download PFAM HMM.

    Download HMM for a given PFAM id.
    File is saved in folder at input
    path.

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
    # Download PFAM HMM
    path_pfam_hmm = f"http://pfam.xfam.org/family/{pfam_id}/hmm"
    try:
        subprocess.run(
            ["curl", "-o", path / f"hmm_{pfam_id}", path_pfam_hmm],
            check=True,
        )
        if (path / f"hmm_{pfam_id}").is_file():
            return True
    except:  # pylint: disable=bare-except
        return False
    return False


def identity(seq1: str, *args: str) -> float:
    """
    Compute sequence identity between two sequences of identical length.

    Parameters
    ----------
    seq1 : str
        Sequence 1
    *args : str
        Sequence 2

    Raises
    ------
    ValueError
        if sequences have different length
    """
    seq1 = "".join(seq1)
    seq2 = args[0]
    if len(seq1) != len(seq2):
        raise ValueError("Sequences should have same length.")
    return sum((i1 == i2 for i1, i2 in zip(seq1, seq2))) / len(seq1)


def read(path: Path) -> SequenceModel:
    """Read a SequenceModel instance written in the input file.

    Read and initialize a new
    SequenceModel instance loaded
    from the input file.

    Parameters
    ----------
    path : Path
        Path to the file where
        the SequenceModel object
        is stored

    Returns
    -------
    SequenceModel
        SequenceModel instance
    """
    file_content = np.load(path)
    return SequenceModel(
        name=path.stem,
        ref_sequence=str(file_content["reference_sequence"]),
        model=Dca(
            h_matrix=file_content["h"], j_matrix=file_content["j"]
        ),
        msa=Msa(
            array=file_content["msa_array"],
            origin=str(file_content["msa_type"]),
        ),
    )
