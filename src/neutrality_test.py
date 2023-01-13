# -*- coding: utf-8 -*-
"""This module contains a class to test for selection using non-synonymous mutation scores.

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
from typing import List, Union, Dict
from pathlib import Path
import ot  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from src.gene import read as read_gene  # type: ignore   # pylint: disable=import-error
from src.gene import Gene  # type: ignore   # pylint: disable=import-error
from src.evolutionary_model import NucleotideSubstitution, JC69  # type: ignore   # pylint: disable=import-error
from src.config import N_poly, N_CI, bin_size  # type: ignore # pylint: disable=import-error


class Score:
    """Store selective pression test results."""

    def __init__(
        self,
        values: Union[List[float], None] = None,
        edges: Union[np.ndarray, None] = None,
        obs_distributions: Union[List[np.ndarray], None] = None,
        all_distribution: Union[np.ndarray, None] = None,
    ):
        """
        Initialize a score instance.

        Attributes
        ----------
        values : List[float], optional
            Score value list,
            default is empty list
        obs_distributions : np.ndarray, optional
            Histograms of DCA scores
            of observed mutations
            default is empty array
        all_distribution : np.ndarray, optional
            Histogram of DCA scores
            of all possible mutations
            default is empty array
        """
        self.values = values
        self.edges = edges
        self.obs_distributions = obs_distributions
        self.all_distribution = all_distribution

    def write(self, output_folder: Path, name: str) -> None:
        """Write Score instance to a .npz file.

        Write Score instance to a .npz file
        in output_folder using input name
        as file name.

        Parameters
        ----------
        output_folder: Path
            Path to output folder
        name: str
            File name

        Returns
        -------
        NoneType
            None
        """
        np.savez(
            output_folder / f"{name}.npz",
            values=self.values,
            edges=self.edges,
            obs_distributions=self.obs_distributions,
            all_distribution=self.all_distribution,
        )


class Test:
    """Quantify selective pressure acting on a gene.

    Attributes
    ----------
    N_poly : int
        Minimal number of mutations
    N_CI : int
        Number of simulations
    bin_size : float
        Bin size for histograms
    evolutionary_model : NucleotideSubstitution
        Model used to simulate mutations
    """

    N_poly: int = N_poly
    N_CI: int = N_CI
    bin_size: float = bin_size
    evolutionary_model: NucleotideSubstitution = JC69()

    def __init__(
        self,
        gene: Gene,
        model_type: str,
        mutation_df: pd.DataFrame,
        test_scores: Union[Dict[str, Score], None] = None,
    ) -> None:
        """
        Initialize a new Test instance.

        Parameters
        ----------
        gene : Gene
            Gene instance
        model_type : str
            Scoring model type (IND
            or DCA)
        mutation_df: pd.DataFrame
            Dataframe of mutations
            ("Locus","AA1","AA2")
        score: Score
            Score of observed
            mutations
        neutral_scores: Score
            Scores of simulated
            mutations
        """
        self.gene = gene
        self.model_type = model_type
        self.mutation_df = mutation_df
        if test_scores is not None:
            self.score = test_scores["obs_score"]
            self.neutral_scores = test_scores["neutral_scores"]
        else:
            self.score = Score()
            self.neutral_scores = Score()

    def compute_test(self) -> None:
        """
        Filter mutations and run test.

        Raises
        ------
        ValueError
            If self.model_type is not a
            key of self.gene.score
        """
        if self.model_type not in self.gene.scores.keys():
            raise ValueError(
                "model_type argument should be a key of gene.scores"
            )
        self.mutation_df = filter_mutation_df(
            self.gene, self.mutation_df, self.model_type
        )
        n_mut = len(self.mutation_df)
        if n_mut >= self.N_poly:
            self.score = compute_score(
                self.gene,
                [self.mutation_df],
                self.model_type,
                self.bin_size,
            )
            loci = self.gene.get_loci(self.model_type)
            self.neutral_scores = compute_score(
                self.gene,
                [
                    self.evolutionary_model.generate_mutations(
                        self.gene.dna_seq, n_mut, loci
                    )
                    for i in range(self.N_CI)
                ],
                self.model_type,
                self.bin_size,
            )

    def write(self, path: Path) -> None:
        """Write Test instance to a folder.

        Writes Test instance to a folder
        using self.gene.name and self.model_type
        for folder name.

        Parameters
        ----------
        output_folder: Path
            Path to output folder

        Returns
        -------
        NoneType
            None
        """
        folder = path / f"{self.gene.name}-{self.model_type}"
        folder.mkdir(parents=True, exist_ok=True)
        self.gene.write(folder)
        self.score.write(folder, "obs_score")
        self.mutation_df.to_csv(folder / "mutations.csv", index=None)
        self.neutral_scores.write(folder, "neutral_scores")


def filter_mutation_df(
    gene: Gene, mutation_df: pd.DataFrame, model_type: str
) -> pd.DataFrame:
    """
    Initialize a new Test instance.

    Parameters
    ----------
    gene : Gene
        Gene instance
    mutation_df : pd.DataFrame
        Dataframe of mutations
        ("Locus","AA1","AA2")
    model_type : str
        Scoring model type (IND
        or DCA)

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe
        filtered on loci covered by
        the model

    Raises
    ------
    ValueError
        If model_type is not a key
        of gene.scores
    """
    if model_type not in gene.scores.keys():
        raise ValueError(
            "model_type argument should be a key of gene.scores"
        )
    loci = gene.get_loci(model_type)
    return mutation_df[mutation_df["Locus"].isin(loci)]


def compute_histogram_bins(
    min_: float, max_: float, bin_size_: float
) -> np.ndarray:
    """Compute bin edges.

    Parameters
    ----------
    min_ : float
        Minimal value
    max_ : float
        Maximal value
    bin_size_ : float
        Bin size

    Returns
    -------
    np.ndarray
        Bin edges
    """
    start = min_ - bin_size_ / 2
    stop = max_ + bin_size_ / 2
    return np.array(
        [
            start + i * bin_size_
            for i in range(int((stop - start) // bin_size_) + 2)
        ]
    )


def transport_cost(
    transport_matrix: np.ndarray, nb_items: int
) -> float:
    """Compute normalized cost from transport matrix.

    Parameters
    ----------
    matrix : float
        Optimal transport matrix
    nb_items : int
        Number of items in histograms

    Returns
    -------
    float
        Normalized transport cost
    """
    left, right = 0, 0
    for i in range(len(transport_matrix)):
        for j in range(len(transport_matrix)):
            if i > j:
                left += transport_matrix[i, j] * (i - j)
            if j > i:
                right += transport_matrix[i, j] * (j - i)
    return (left - right) / nb_items


def compute_optimal_transport(
    hist_obs: np.ndarray,
    hist_all: np.ndarray,
    matrix: np.ndarray,
    bin_size__: float,
) -> float:
    """Compute cost for optimal transport between both distributions.

    Parameters
    ----------
    hist_obs : np.ndarray
        Histogram of observed
        mutation scores
    hist_all : np.ndarray
        Histogram of all possible
        mutation scores
    matrix : float
        Optimal transport matrix

    Returns
    -------
    float
        Normalized transport cost
    """
    lowest_common_multiple = np.lcm(hist_all.sum(), hist_obs.sum())
    hist_all_norm, hist_obs_norm = (
        hist_all * (lowest_common_multiple / hist_all.sum()),
        hist_obs * (lowest_common_multiple / hist_obs.sum()),
    )
    return (
        transport_cost(
            ot.emd(hist_all_norm, hist_obs_norm, matrix),
            lowest_common_multiple,
        )
        * bin_size__
    )


def compute_score(
    gene: Gene,
    mutation_df_list: List[pd.DataFrame],
    model_type: str,
    bin_size_: float,
) -> Score:
    """
    Compute scores for a gene.

    Compute selective pressure score
    for each serie of mutations observed
    on the input gene.

    Parameters
    ----------
    gene : Gene
        Gene instance
    mutation_df_list : pd.DataFrame
        List of dataframes of mutations
        ("Locus","AA1","AA2")
    model_type : str
        Scoring model type (IND
        or DCA)
    bin_size_ : float
        Bin size for histograms

    Returns
    -------
    score : Score
        Selective pressure scores
    """
    all_mutations_df = gene.get_all_mutations(model_type)
    all_mutations_df = gene.compute_mutation_scores(
        all_mutations_df, model_type
    )
    edges = compute_histogram_bins(
        all_mutations_df["Score"].min(),
        all_mutations_df["Score"].max(),
        bin_size_,
    )
    bins_nb = len(edges) - 1
    matrix = ot.dist(
        np.arange(bins_nb).reshape((bins_nb, 1)),
        np.arange(bins_nb).reshape((bins_nb, 1)),
    ).astype(float)
    matrix /= matrix.max()
    hist_all = np.histogram(
        all_mutations_df["Score"], bins=edges, density=False
    )[0]
    values = list()
    obs_distributions = list()
    for mutation_df in mutation_df_list:
        if "Score" not in mutation_df.columns:
            mutation_df = gene.compute_mutation_scores(
                mutation_df, model_type
            )
        hist_obs = np.histogram(
            mutation_df["Score"], bins=edges, density=False
        )[0]
        values.append(
            compute_optimal_transport(
                hist_obs, hist_all, matrix, bin_size_
            )
        )
        obs_distributions.append(hist_obs)
    return Score(
        values=values,
        edges=edges,
        obs_distributions=obs_distributions,
        all_distribution=hist_all,
    )


def read_score(path: Path) -> Score:
    """Read a Score instance written in the input file.

    Read and initialize a new
    Score instance loaded from
    the input file.

    Parameters
    ----------
    path : Path
        Path to the file where
        the Score object is
        stored

    Returns
    -------
    Score
        Score instance
    """
    file_content = np.load(path, allow_pickle=True)
    return Score(
        values=file_content["values"],
        edges=file_content["edges"],
        obs_distributions=file_content["obs_distributions"],
        all_distribution=file_content["all_distribution"],
    )


def read(path: Path) -> Test:
    """Read a Test instance written in the input folder.

    Read and initialize a new
    Test instance loaded from
    the input folder.

    Parameters
    ----------
    path : Path
        Path to the folder where
        the Test object is
        stored

    Returns
    -------
    Test
        Test instance
    """
    gene_name = path.stem.split("-")[0]
    model_type = path.stem.split("-")[1]
    gene = read_gene(path / f"{gene_name}")
    obs_score = read_score(path / "obs_score.npz")
    neutral_scores = read_score(path / "neutral_scores.npz")
    mutation_df = pd.read_csv(path / "mutations.csv")
    return Test(
        gene,
        model_type,
        mutation_df,
        {"obs_score": obs_score, "neutral_scores": neutral_scores},
    )
