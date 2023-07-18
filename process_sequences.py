# -*- coding: utf-8 -*-
"""This module allows to process input DNA sequences to generate a clean dataset."""

from typing import List, Dict, Tuple
from pathlib import Path
import subprocess
import argparse
from joblib import Parallel, delayed  # type: ignore
from Bio import Phylo  # type: ignore
from Bio.Seq import Seq  # type: ignore
from Bio.SeqRecord import SeqRecord  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from src.bio_basics import (  # type: ignore # pylint: disable=import-error
    read_fasta,
    write_fasta,
    read_fasta_dict,
    translate,
    reverse_aln,
    generate_codons_df,
    codon_to_aa_df,
    pivot_codons_df,
)
from src.config import get_path  # type: ignore # pylint: disable=import-error

from src.gene import to_mutation_df, initialize_protein  # type: ignore # pylint: disable=import-error
from src.neutrality_test import Test  # type: ignore # pylint: disable=import-error


def extract_unique_seq(path: Path) -> Tuple[List, pd.DataFrame]:
    """Extract unique sequences from fasta file.

    Read sequences contained in
    input fasta file, remove duplicates
    and assign them a new id. Return list
    of unique sequences with new id and
    dataframe with correspondance between
    old and new ids.

    Parameters
    ----------
    path : str, pathlib.Path
        Path to the input fasta file

    Returns
    -------
    List
        List of unique sequences contained
        in the fasta file
    pandas.DataFrame
        Dataframe (Id, Unique_Id)
    """
    sequences = read_fasta(path)
    content: Dict[str, List] = {"Id": [], "Seq": []}
    for seq in sequences:
        content["Id"].append(seq.id)
        content["Seq"].append(str(seq.seq))
    sequences_df = pd.DataFrame(content)
    unique_sequences_df = (
        sequences_df.groupby("Seq")
        .size()
        .reset_index()
        .sort_values(by=0, ascending=False)[["Seq"]]
    )
    unique_sequences_df["Unique_Id"] = [
        str(i) for i in range(len(unique_sequences_df))
    ]
    sequences_df = sequences_df.merge(unique_sequences_df, on="Seq")[
        ["Id", "Unique_Id"]
    ]
    unique_sequences = [
        SeqRecord(
            Seq(getattr(row, "Seq")),
            id=(getattr(row, "Unique_Id")),
            description="",
        )
        for row in unique_sequences_df.itertuples()
    ]
    return unique_sequences, sequences_df


def add_outgroup(
    path: Path, outgroup_file: Path, hit_id: str = ""
) -> Tuple[List, pd.DataFrame]:
    """Extract unique DNA sequences and add outgroup.

    Extract unique DNA sequences from input
    fasta file. Run Blast to find an outgroup
    sequence in outgroup_file database.

    Parameters
    ----------
    path : pathlib.Path
        Path to input fasta file
    outgroup_file : pathlib.Path
        Path to outgroup Blast
        database
    hit_id : str, optional
        Outgroup hit id, if empty
        string is given, will retrieve
        it using Blast search.

    Returns
    -------
    List
        List of unique DNA sequences
    pandas.DataFrame
        Dataframe of correspondance between
        sequence ids.
    """
    tmp_folder = get_path("tmp")
    unique_sequences, sequences_df = extract_unique_seq(path)
    write_fasta([unique_sequences[0]], tmp_folder / path.name)
    outgroup_seq = read_fasta_dict(outgroup_file)
    outgroup_hit = SeqRecord(
        outgroup_seq[hit_id].seq, id="Outgroup", description=""
    )
    unique_sequences = unique_sequences + [outgroup_hit]
    write_fasta(unique_sequences, tmp_folder / f"{path.stem}_DNA.fasta")
    return unique_sequences, sequences_df


def translate_sequences(path: Path) -> List:
    """Translate DNA sequences into AA sequences.

    Translate DNA sequence in input file into
    AA sequences that are cropped at first stop
    codon.

    Parameters
    ----------
    path : pathlib.Path
        Path to input fasta file

    Returns
    -------
    List
        List of translated sequences
    """
    return [
        SeqRecord(
            Seq(translate(str(seq.seq)).split("*")[0]),
            id=seq.id,
            description="",
        )
        for seq in read_fasta(path)
    ]


def aln_sequences(input_file: Path, output_file: Path) -> None:
    """Align sequences using mafft.

    Parameters
    ----------
    input_file : pathlib.Path
        Fasta file with sequences
        to align
    output_file : pathlib.Path
        Fasta file where aligned
        sequences will be written

    Returns
    -------
    NoneType
        None
    """
    call = subprocess.run(
        ["mafft", "--retree", "1", "--quiet", input_file],
        check=True,
        stdout=subprocess.PIPE,
    )
    with open(output_file, "w") as output:
        output.write(call.stdout.decode("utf-8"))


def extract_cds(dna_seq: str) -> str:
    """Extract coding sequence up to the first stop codon.

    Parameters
    ----------
    dna_seq : str
        DNA sequence

    Returns
    -------
    str
        Coding sequence
    """
    aa_seq = translate(dna_seq).split("*")[0]
    return dna_seq[: len(aa_seq) * 3]


def reverse_aln_sequences(
    input_dna_file: Path, input_aa_file: Path, output_file: Path
) -> None:
    """Reverse align DNA sequences from AA sequence alignment.

    Reverse align DNA sequences in input_dna_file
    file using AA sequence alignment in input_aa_file
    file and write resulting alignment in output_file.
    Remove stop codons and subsequent codons.

    Parameters
    ----------
    input_dna_file : pathlib.Path
        Fasta file with raw
        DNA sequences
    input_aa_file : pathlib.Path
        Fasta file with aligned
        amino acid sequences
    output_file : pathlib.Path
        Path where output file
        will be written

    Returns
    -------
    NoneType
        None
    """
    raw_dna_seq_dict = read_fasta_dict(input_dna_file)
    aln_aa_seq_dict = read_fasta_dict(input_aa_file)
    aln_dna_seq = list()
    for aa_id in aln_aa_seq_dict.keys():
        if aa_id in raw_dna_seq_dict:
            cds = extract_cds(str(raw_dna_seq_dict[aa_id].seq))
        else:
            seq_len = len(
                str(aln_aa_seq_dict[aa_id].seq).replace("-", "")
            )
            cds = "N" * 3 * seq_len
        aln_dna_seq.append(
            SeqRecord(
                Seq(reverse_aln(cds, str(aln_aa_seq_dict[aa_id].seq))),
                id=aa_id,
                description="",
            )
        )
    write_fasta(aln_dna_seq, output_file)


def aln_on_seq(path: Path, seq_id: str) -> None:
    """Align sequences on seq_id sequence by removing inserts.

    Read sequences in fasta file at path,
    extract seq_id and remove each character
    which is gapped in seq_id in the overall
    alignment (insert). Remove reference
    sequence from alignment and re-write
    input_file.

    Parameters
    ----------
    path : pathlib.Path
        Path to input fasta file
    seq_id : str
        Id of reference sequence
        in fasta file

    Returns
    -------
    NoneType
        None
    """
    seq_dict = read_fasta_dict(path)
    reference = str(seq_dict[seq_id].seq)
    del seq_dict[seq_id]
    processed_sequences = [
        SeqRecord(
            Seq(
                "".join(
                    [
                        character
                        for character, i in zip(str(seq.seq), reference)
                        if i != "-"
                    ]
                )
            ),
            id=seq.id,
            description="",
        )
        for seq in seq_dict.values()
    ]
    write_fasta(processed_sequences, path)


def get_unique_id(sequences_df: pd.DataFrame, seq_id: str) -> str:
    """Return unique id of sequence with id=seq_id.

    Parameters
    ----------
    sequences_df : pandas.DataFrame
        DataFrame with correspondance
        between ids and unique sequence
        ids
    seq_id : str
        Input sequence id

    Returns
    -------
    str
        Unique sequence id
    """
    return list(
        sequences_df[sequences_df["Id"] == seq_id]["Unique_Id"]
    )[0]


def rename_seq(sequences: List, old_id: str, new_id: str) -> List:
    """Modify sequence id.

    Parameters
    ----------
    sequences : List
        List of sequences in
        SeqRecord format
    old_id : str
        Id to change
    new_id : str
        New id

    Returns
    -------
    List
        List of sequences in
        SeqRecord format with
        modified id
    """
    new_sequences = list()
    for seq in sequences:
        if seq.id == old_id:
            new_sequences.append(
                SeqRecord(seq.seq, id=new_id, description="")
            )
        else:
            new_sequences.append(seq)
    return new_sequences


def name_nodes(tree: Phylo.Newick.Tree) -> None:
    """Name internal nodes of the tree in place.

    Parameters
    ----------
    tree : Phylo.Newick.Tree
        Input tree

    Returns
    -------
    NoneType
        None
    """
    if tree.clade.clades[0].name is None:
        tree.clade.clades[0].name = "Root"
    for index, clade in enumerate(tree.get_nonterminals()):
        clade.confidence = None
        if clade.name is None and clade.branch_length is not None:
            clade.name = f"Internal_node_{index}"
    for clade in tree.get_terminals():
        clade.confidence = None


def process_file(
    fasta_file: Path,
    outgroup_file: Path,
    reference_sequences_dict: Dict,
    hit_id: str = "",
) -> None:
    """Compute sequence tree for DNA sequences with FastTree.

    Parameters
    ----------
    fasta_file : pathlib.Path
        File with input DNA sequences
        to process
    outgroup_file : pathlib.Path
        Path to outgroup Blast database
    hit_id : str, optional
        Outgroup hit id, if empty
        string is given, will retrieve
        it using Blast search.

    Returns
    -------
    NoneType
        None
    """
    tmp_folder = get_path("tmp")
    aln_seq_folder = get_path("msa")
    seq_id_folder = get_path("seq_id")
    ref_sequence = SeqRecord(
        reference_sequences_dict[fasta_file.stem].seq,
        id="Reference",
        description="",
    )
    unique_sequences, sequences_df = add_outgroup(
        fasta_file, outgroup_file, hit_id
    )
    if len(unique_sequences) > 0:
        aa_sequences = translate_sequences(
            tmp_folder / f"{fasta_file.stem}_DNA.fasta"
        )
        write_fasta(
            aa_sequences + [ref_sequence],
            tmp_folder / f"{fasta_file.stem}_AA.fasta",
        )
        aln_sequences(
            tmp_folder / f"{fasta_file.stem}_AA.fasta",
            tmp_folder / f"{fasta_file.stem}_AA_aln.fasta",
        )
        reverse_aln_sequences(
            tmp_folder / f"{fasta_file.stem}_DNA.fasta",
            tmp_folder / f"{fasta_file.stem}_AA_aln.fasta",
            tmp_folder / f"{fasta_file.stem}_DNA_aln.fasta",
        )
        (tmp_folder / f"{fasta_file.stem}_AA.fasta").unlink()
        (tmp_folder / f"{fasta_file.stem}_AA_aln.fasta").unlink()
        (tmp_folder / f"{fasta_file.stem}_DNA.fasta").unlink()
        aln_on_seq(
            tmp_folder / f"{fasta_file.stem}_DNA_aln.fasta", "Reference"
        )
        aln_unique_sequences, aln_sequences_df = extract_unique_seq(
            tmp_folder / f"{fasta_file.stem}_DNA_aln.fasta"
        )
        (tmp_folder / f"{fasta_file.stem}_DNA_aln.fasta").unlink()
        outgroup_id = get_unique_id(aln_sequences_df, "Outgroup")
        if (
            len(aln_sequences_df[aln_sequences_df["Unique_Id"] == outgroup_id])
            == 1
        ):
            write_fasta(
                rename_seq(
                    aln_unique_sequences, outgroup_id, "Outgroup"
                ),
                aln_seq_folder / fasta_file.name,
            )
            aln_sequences_df = aln_sequences_df[
                aln_sequences_df["Unique_Id"] != outgroup_id
            ].merge(
                sequences_df,
                left_on=["Id"],
                right_on=["Unique_Id"],
                suffixes=["_aln", "_raw"],
            )[
                ["Id_raw", "Unique_Id_aln"]
            ]
            aln_sequences_df.columns = ["Sequence_Id", "Unique_Id"]
            aln_sequences_df.to_csv(
                seq_id_folder / fasta_file.with_suffix(".csv").name,
                index=None,
            )


def compute_tree(input_file: Path, output_file: Path) -> None:
    """Compute sequence tree for DNA sequences with FastTree.

    Parameters
    ----------
    input_file : pathlib.Path
        File with DNA sequence
        alignment in fasta format
    output_file : pathlib.Path
        Path where tree will be
        written in newick format.

    Returns
    -------
    NoneType
        None
    """
    tmp_folder = get_path("tmp")
    call = subprocess.run(
        ["fasttree", "-quiet", "-nt", input_file],
        check=True,
        stdout=subprocess.PIPE,
    )
    with open(tmp_folder / output_file.name, "w") as output:
        output.write(call.stdout.decode("utf-8"))
    tree = Phylo.read(tmp_folder / output_file.name, "newick")
    tree.root_with_outgroup({"name": "Outgroup"})
    name_nodes(tree)
    with open(output_file, "w") as output:
        Phylo.write(tree, output, "newick")
    (tmp_folder / output_file.name).unlink()


def infer_ancestral_sequence(input_fasta_file: Path) -> None:
    """Run iqtree to reconstruct ancestral sequences (codon model).

    Parameters
    ----------
    input_fasta_file : pathlib.Path
        File with DNA sequences

    Returns
    -------
    NoneType
        None
    """
    tree_folder = get_path("tree")
    output_folder = get_path("ancestral")
    try:
        subprocess.run(
            [
                "iqtree",
                "-s",
                input_fasta_file,
                "-te",
                tree_folder / input_fasta_file.with_suffix(".nhw").name,
                "-o",
                "Outgroup",
                "-st",
                "CODON",
                "-m",
                "GY",
                "-asr",
                "-quiet",
            ],
            check=True,
        )
        Path(f"{input_fasta_file}.state").rename(
            output_folder / input_fasta_file.with_suffix(".csv").name
        )
        Path(f"{input_fasta_file}.ckp.gz").unlink()
        Path(f"{input_fasta_file}.iqtree").unlink()
        Path(f"{input_fasta_file}.log").unlink()
        Path(f"{input_fasta_file}.treefile").unlink()
        return None
    except:  # pylint: disable=bare-except
        return None


def compute_ancestral_df(
    ancestral_seq_file: Path, node: str = "Root"
) -> pd.DataFrame:
    """Return codon dataframe of ancestral sequence.

    Parse iqtree output .state file to
    extract the sequence corresponding
    to the input node id and return it
    as a codon dataframe.

    Parameters
    ----------
    ancestral_seq_file : pathlib.Path
        iqtree output .state file
    node : str, optional
        Node id for which the sequence
        will be extracted, default is
        "Root"

    Returns
    -------
    pandas.DataFrame
        Codon dataframe of ancestral
        sequence
    """
    ancestral_df = pd.read_csv(ancestral_seq_file, skiprows=8, sep="\t")
    ancestral_df["Site"] = ancestral_df["Site"].astype(int)
    ancestral_df = ancestral_df[
        ancestral_df["Node"] == node
    ].sort_values(by="Site")[["Site", "State"]]
    ancestral_df.columns = ["Locus", "Codon"]
    return ancestral_df


def gather_all_mutations(
    seq_id_file: Path, fasta_file: Path, root_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Gather all derived and ancestral codons.

    Parse sequences in input fasta file to
    extract observed codons and distinguish
    wild-type from derived states using the
    reference root_matrix codon matrix.

    Parameters
    ----------
    seq_id_file : pathlib.Path
        Input csv file of correspondance
        between sequence id and unique id
    fasta_file : pathlib.Path
        Input fasta file of DNA sequences
    root_matrix : pandas.DataFrame
        Root matrix

    Returns
    -------
    pandas.DataFrame
        Mutation dataframe
    """
    codons_matrix_coef = pd.DataFrame(
        np.zeros(root_matrix.shape),
        columns=root_matrix.columns,
        index=root_matrix.index,
    )
    codons_matrix_unique = pd.DataFrame(
        np.zeros(root_matrix.shape),
        columns=root_matrix.columns,
        index=root_matrix.index,
    )
    seq_df = (
        pd.read_csv(seq_id_file)
        .groupby("Unique_Id")
        .size()
        .reset_index()
        .rename(columns={0: "Coef"})
    )
    seq_df["Unique_Id"] = seq_df["Unique_Id"].astype(str)
    seq_dict = read_fasta_dict(fasta_file)
    for row in seq_df.itertuples():
        codons_df = generate_codons_df(
            str(seq_dict[getattr(row, "Unique_Id")].seq)
        )
        codons_matrix = pivot_codons_df(codons_df)
        codons_matrix_coef = codons_matrix_coef.add(
            codons_matrix * getattr(row, "Coef"), fill_value=0
        )
        codons_matrix_unique = codons_matrix_unique.add(
            codons_matrix, fill_value=0
        )
    mut_matrix_coef = codon_to_aa_df(codons_matrix_coef).astype(int)
    mut_matrix_unique = codon_to_aa_df(codons_matrix_unique).astype(int)
    native_matrix = codon_to_aa_df(root_matrix).astype(int)
    mut_df_coef = to_mutation_df(
        (
            (
                (
                    (mut_matrix_coef > 0).astype(int) != native_matrix
                ).astype(int)
                - native_matrix
            )
            * mut_matrix_coef
        ).reset_index()
    )
    mut_df_unique = to_mutation_df(
        (
            (
                (
                    (mut_matrix_unique > 0).astype(int) != native_matrix
                ).astype(int)
                - native_matrix
            )
            * mut_matrix_unique
        ).reset_index()
    )
    return mut_df_coef.merge(
        mut_df_unique,
        on=["Locus", "AA1", "AA2"],
        suffixes=["_all", "_unique"],
    )[
        [
            "Locus",
            "AA1",
            "AA2",
            "Count1_all",
            "Count2_all",
            "Total_all",
            "Count1_unique",
            "Count2_unique",
            "Total_unique",
        ]
    ]


def initialize_test(
    protein_models: List[Path], gather: bool = False
) -> None:
    """Create, run and write test instance.

    Parameters
    ----------
    protein_models : List[pathlib.Path]
        Paths to model files
    gather : bool, optional
        Should all protein_models be gathered
        in a single Test instance, default is
        False

    Returns
    -------
    NoneType
        None
    """
    protein_name = protein_models[0].stem.split("_")[0]
    if (get_path("ancestral") / f"{protein_name}.csv").is_file():
        root_df = compute_ancestral_df(
            get_path("ancestral") / f"{protein_name}.csv"
        )
        dna_seq = "".join(root_df.sort_values("Locus")["Codon"])
        if gather:
            iterate_over = zip([protein_name], [protein_models])
        else:
            iterate_over = zip(
                [file.stem for file in protein_models],
                [[model] for model in protein_models],
            )
        for name, models in iterate_over:
            gene = initialize_protein(name, dna_seq, models)
            mut_df = gather_all_mutations(
                get_path("seq_id") / f"{protein_name}.csv",
                get_path("msa") / f"{protein_name}.fasta",
                pivot_codons_df(root_df),
            )
            for model_type in gene.scores.keys():
                if model_type=="DCA":
                    test = Test(
                        gene,
                        model_type,
                        gene.compute_mutation_scores(mut_df, model_type),
                    )
                    test.compute_test()
                    test.write(get_path("test"))


def select_models() -> Dict[str, List[Path]]:
    """List all available models and corresponding proteins.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary {protein_name:protein_models list}
    """
    models_folder = get_path("dca")
    aln_seq_folder = get_path("msa")
    models: Dict[str, List[Path]] = dict()
    for file in models_folder.iterdir():
        if file.suffix == ".npz":
            protein_name = file.stem.split("_")[0]
            if (aln_seq_folder / f"{protein_name}.fasta").is_file():
                if protein_name not in models.keys():
                    models[protein_name] = []
                models[protein_name].append(file)
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consensus", help="input fasta file of consensus sequences"
    )
    parser.add_argument(
        "--outgroup", help="input fasta file of outgroup sequences"
    )
    parser.add_argument(
        "--outgroup_hits",
        help="input dataframe of correspondance between consensus and outgroup sequences",
    )
    parser.add_argument(
        "--sequences_dir",
        help="input folder of fasta files of species gene sequences",
    )
    parser.add_argument(
        "--nthreads", help="number of threads for parallelization"
    )
    args = parser.parse_args()
    ref_seq_file = Path(args.consensus)
    nthreads = int(args.nthreads) if args.nthreads is not None else 1
    outgroup_hits = (
        pd.read_csv(args.outgroup_hits)
        if args.outgroup_hits is not None
        else None
    )
    get_path("tmp").mkdir(parents=True, exist_ok=True)
    get_path("tree").mkdir(parents=True, exist_ok=True)
    get_path("seq_id").mkdir(parents=True, exist_ok=True)
    get_path("msa").mkdir(parents=True, exist_ok=True)
    get_path("ancestral").mkdir(parents=True, exist_ok=True)
    get_path("test").mkdir(parents=True, exist_ok=True)
    filenames = [
        file
        for file in Path(args.sequences_dir).iterdir()
        if file.suffix == ".fasta"
    ]
    reference_sequences = read_fasta_dict(ref_seq_file)
    hits_dict = {
        row["consensus"]: row["outgroup"]
        for _, row in outgroup_hits.iterrows()
    }
    Parallel(n_jobs=nthreads)(
        delayed(process_file)(
            fasta_file,
            Path(args.outgroup),
            reference_sequences,
            hits_dict[fasta_file.stem],
        )
        for fasta_file in filenames
        if fasta_file.stem in hits_dict.keys()
    )
    Parallel(n_jobs=nthreads)(
        delayed(compute_tree)(
            fasta_file, get_path("tree") / f"{fasta_file.stem}.nhw"
        )
        for fasta_file in get_path("msa").iterdir()
        if fasta_file.suffix == ".fasta"
    )
    Parallel(n_jobs=nthreads)(
        delayed(infer_ancestral_sequence)(fasta_file)
        for fasta_file in get_path("msa").iterdir()
        if fasta_file.suffix == ".fasta"
    )
    trained_models = select_models()
    Parallel(n_jobs=nthreads)(
        delayed(initialize_test)(protein_models)
        for protein_models in trained_models.values()
    )
