# -*- coding: utf-8 -*-
"""This module creates and trains DCA models."""

from pathlib import Path
import argparse

from joblib import Parallel, delayed  # type: ignore
from Bio import SeqIO  # type: ignore
from src.config import data_folder, models_folder, tmp_folder  # type: ignore
from src.sequence_model import SequenceModel  # type: ignore


def create_protein_model(
    seq_id: str,
    reference_sequence: str,
    tmp_path: Path,
    output_path: Path,
) -> None:
    """Create and train a SequenceModel instance.

    Parameters
    ----------
    seq_id : str
        SequenceModel id
    reference_sequence : str
        Reference sequence
    tmp_path : pathlib.Path
        Temporary folder
    output_path : pathlib.Path
        Output folder where the SequenceModel
        instance will be written

    Returns
    -------
    SequenceModel
        SequenceModel instance
    """
    model = SequenceModel(f"{seq_id}", ref_sequence=reference_sequence)
    model.gather_msa(tmp_path)
    model.train_dca(tmp_path)
    model.write(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="input fasta file of amino-acid sequences"
    )
    parser.add_argument(
        "--nthreads", help="number of threads for parallelization"
    )
    args = parser.parse_args()
    input_fasta = Path(args.input)
    nthreads = int(args.nthreads) if args.nthreads is not None else 1

    data_folder.mkdir(exist_ok=True, parents=True)
    models_folder.mkdir(exist_ok=True, parents=True)
    tmp_folder.mkdir(exist_ok=True, parents=True)

    sequences = {
        seq.id: str(seq.seq)
        for seq in SeqIO.parse(input_fasta, "fasta")
    }

    Parallel(n_jobs=nthreads)(
        delayed(create_protein_model)(
            seq_id, sequences[seq_id], tmp_folder, models_folder
        )
        for seq_id in sequences.keys()
    )
