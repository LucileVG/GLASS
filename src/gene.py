# -*- coding: utf-8 -*-
"""This module contains a class to manipulate and store genes.

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
from typing import Dict, List, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from src.config import AAs, nucleotides  # type: ignore # pylint: disable=import-error
from src import sequence_model  # type: ignore # pylint: disable=import-error
from src.bio_basics import translate  # type: ignore # pylint: disable=import-error


class Gene:
    """Store and manipulate a gene sequence and its non-synonymous single mutants.

    Gather a gene and model mutation
    effect by combining DCA and IND models.

    Attributes
    ----------
    valid_AAs : List[str]
        Valid characters for amino acids.

    TODO
    """

    valid_AAs: List[str] = AAs
    valid_nucleotides: List[str] = nucleotides

    def __init__(
        self, name: str, dna_seq: str, scores: Dict[str, pd.DataFrame]
    ):
        """
        Initialize a new Gene instance.

        Parameters
        ----------
        name : str
            gene name
        dna_seq : str
            reference DNA sequence
        scores : dict
            Dictionary of single
            mutant scores.
        """
        self.name = name
        self.dna_seq = dna_seq
        self.scores = scores

    def write(self, output_folder: Path) -> None:
        """Write Gene instance to a .npz file.

        Writes Gene instance to a .npz
        file using the name attribute
        as file name.

        Parameters
        ----------
        output_folder: Path
            Path to output folder

        Returns
        -------
        NoneType
            None
        """
        gene_folder = output_folder / f"{self.name}"
        gene_folder.mkdir(parents=True, exist_ok=True)
        with open(gene_folder / "dna_seq.txt", "w") as seq:
            seq.write(self.dna_seq)
        for key in self.scores.keys():
            self.scores[key].to_csv(
                gene_folder / f"{key}.csv", index=None
            )

    def get_loci(self, model_type: str) -> List[int]:
        """
        Return amino acid loci covered by model.

        Parameters
        ----------
        model_type : str
            Model used to score mutations
            ("IND" or "DCA")

        Returns
        -------
        List[int]
            List of amino acid loci

        Raises
        ------
        ValueError
            self.scores does not contain
            model_type key
        """
        if model_type not in self.scores.keys():
            raise ValueError(
                "Scores attribute should contain model_type score matrix."
            )
        return list(self.scores[model_type]["Locus"])

    def get_protein_seq(self) -> str:
        """
        Translate gene into protein.

        Translate gene DNA sequence into
        an amino acid sequence that ends at
        the first codon stop encountered.
        Returns an empty string if DNA
        sequence length is not a multiple
        of 3 or if there are invalid
        characters.

        Returns
        -------
        str
            Protein sequence
        """
        return translate(self.dna_seq.upper()).split("*")[0]

    def get_all_mutations(self, model_type: str) -> pd.DataFrame:
        """Return all possible mutations for model_type model.

        Parameters
        ----------
        model_type : str
            Model used to score mutations
            ("IND" or "DCA")

        Returns
        -------
        pd.DataFrame
            Copy of the input dataframe
            with additional mutation score
            column

        Raises
        ------
        ValueError
            self.scores does not contain
            model_type key
        """
        if model_type not in self.scores.keys():
            raise ValueError(
                "Scores attribute should contain model_type score matrix."
            )
        scores = self.scores[model_type].set_index("Locus")
        real_amino_acids = [i for i in scores.columns if i != "-"]
        mutations_matrix = pd.DataFrame(
            np.ones((len(scores), len(real_amino_acids))),
            columns=real_amino_acids,
        )
        mutations_matrix["Locus"] = scores.index
        aa_sequence = self.get_protein_seq()
        for index, row in mutations_matrix.iterrows():
            mutations_matrix.loc[
                index, aa_sequence[int(row["Locus"] - 1)]
            ] = -1
        return to_mutation_df(mutations_matrix.astype(int))

    def compute_mutation_scores(
        self, mutation_df: pd.DataFrame, model_type: str
    ) -> pd.DataFrame:
        """
        Compute mutation scores.

        Return a copy of the input dataframe
        with an additional column containing
        mutation scores according to model_type
        model. Remove mutations that are not
        covered by the model.

        Parameters
        ----------
        mutation_df : pd.DataFrame
            Dataframe of mutations
            ("Locus","AA1","AA2")
        model_type : str
            Model used to score mutations
            ("IND" or "DCA")

        Returns
        -------
        pd.DataFrame
            Copy of the input dataframe
            with additional mutation score
            column

        Raises
        ------
        ValueError
            model_type is not "IND" or "DCA"
        ValueError
            self.scores does not contain
            model_type key
        """
        if model_type not in ["IND", "DCA"]:
            raise ValueError("model_type should be IND or DCA.")
        if model_type not in self.scores.keys():
            raise ValueError(
                "Scores attribute should contain model_type score matrix."
            )
        score_matrix = self.scores[model_type].set_index("Locus")
        mutation_score_df = mutation_df[
            mutation_df["Locus"].isin(score_matrix.index)
        ]
        mutation_score_df["Score"] = mutation_score_df.apply(
            lambda row: score_matrix.loc[row["Locus"], row["AA2"]]
            - score_matrix.loc[row["Locus"], row["AA1"]],
            axis=1,
        )
        return mutation_score_df


def initialize_protein(
    name: str, dna_seq: str, model_files: List[Path]
) -> Union[Gene, None]:
    """
    Combine models to create a new Gene instance.

    Combine models listed in
    the model_files list (in
    decreasing priority order)
    and initialize a new Gene
    instance.

    Parameters
    ----------
    name : str
        gene name
    dna_seq : str
        reference DNA sequence
    model_files : list
        Ordered list of paths
        to model files.

    Returns
    -------
    Gene
        Gene instance
    """
    models = [sequence_model.read(file) for file in model_files]
    if len(models) > 0:
        # Select sites for each model
        # depending on model site
        # coverage and model priority
        # ranking
        coverages = [model.get_coverage() for model in models]
        combined_cov = np.zeros(len(coverages[0]))
        selected_sites = list()
        selected_loci = list()
        for index, cov in enumerate(coverages):
            combined_cov += (
                (cov == 2) * (combined_cov == 0) * (index + 1)
            )
            selected_sites.append(
                [
                    i - 1
                    for i in np.cumsum((cov > 0))
                    * (cov > 0)
                    * (combined_cov == (index + 1))
                    if i > 0
                ]
            )
            selected_loci.append(
                [
                    index + 1
                    for index, value in enumerate(
                        np.cumsum((cov > 0))
                        * (cov > 0)
                        * (combined_cov == (index + 1))
                    )
                    if value > 0
                ]
            )
        # Compute score matrices
        # for each model
        scores: List[List] = [list(), list()]
        for model, sites, loci in zip(
            models, selected_sites, selected_loci
        ):
            score = model.get_ind_scores().loc[sites, :]
            score.index = loci
            scores[0].append(score)
            score = model.get_dca_scores().loc[sites, :]
            score.index = loci
            scores[1].append(score)
        return Gene(
            name=name,
            dna_seq=dna_seq,
            scores={
                "IND": pd.concat(scores[0])
                .sort_index()
                .reset_index()
                .rename(columns={"index": "Locus"}),
                "DCA": pd.concat(scores[1])
                .sort_index()
                .reset_index()
                .rename(columns={"index": "Locus"}),
            },
        )
    return None


def read(path: Path) -> Gene:
    """Read a Gene instance written in the input file.

    Read and initialize a new
    Gene instance loaded from
    the input file.

    Parameters
    ----------
    path : Path
        Path to the file where
        the Gene object is
        stored

    Returns
    -------
    Gene
        Gene instance
    """
    # file_content = np.load(path, allow_pickle=True)
    with open(path / "dna_seq.txt", "r") as file:
        dna_seq = "".join(line for line in file)
    scores = {
        file.stem: pd.read_csv(file)
        for file in path.iterdir()
        if file.suffix == ".csv"
    }
    return Gene(name=path.stem, dna_seq=dna_seq, scores=scores,)


def to_mutation_matrix(
    mutation_df: pd.DataFrame, amino_acids: List[str]
) -> pd.DataFrame:
    """Convert dataframe to matrix.

    Parameters
    ----------
    mutation_df: pd.DataFrame
        Dataframe of mutations
        ("Locus","AA1","AA2")
    amino_acids: List[str]
        List of amino acids
        characters

    Returns
    -------
    pd.DataFrame
        DataFrame storing
        the mutation matrix
    """
    mut_matrix = np.zeros(
        (mutation_df["Locus"].max(), len(amino_acids))
    )
    for row in mutation_df.itertuples():
        mut_matrix[
            getattr(row, "Locus") - 1,
            amino_acids.index(getattr(row, "AA2")),
        ] = 1
        mut_matrix[
            getattr(row, "Locus") - 1,
            amino_acids.index(getattr(row, "AA1")),
        ] = -1
    mut_df = pd.DataFrame(mut_matrix)
    mut_df.columns = amino_acids
    mut_df["Locus"] = [i + 1 for i in range(len(mut_df))]
    return mut_df[np.abs(mut_df[amino_acids]).sum(axis=1) > 0]


def to_mutation_df(mutation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert matrix to dataframe.

    Parameters
    ----------
    mutation_matrix: pd.DataFrame
        Dataframe storing the
        mutation matrix

    Returns
    -------
    pd.DataFrame
        DataFrame of mutations
        ("Locus","AA1","AA2")
    """
    amino_acids = list(mutation_matrix.columns)
    amino_acids.remove("Locus")
    mutation_matrix = mutation_matrix.set_index("Locus")
    mutation_df = (
        mutation_matrix[
            np.abs(mutation_matrix[amino_acids]).sum(axis=1) > 0
        ]
        .stack()
        .reset_index()
        .rename(columns={"level_1": "AA"})
    )
    mutation_df["Count"] = mutation_df[0].apply(np.abs)
    total_df = (
        np.abs(mutation_matrix[amino_acids])
        .sum(axis=1)
        .reset_index()
        .rename(columns={0: "Total"})
    )
    aa1_df = mutation_df[mutation_df[0] < 0][["Locus", "AA", "Count"]]
    aa2_df = mutation_df[mutation_df[0] > 0][["Locus", "AA", "Count"]]
    return aa1_df.merge(aa2_df, on="Locus", suffixes=["1", "2"]).merge(
        total_df, on=["Locus"], how="left"
    )
