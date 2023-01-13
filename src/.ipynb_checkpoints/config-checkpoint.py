from pathlib import Path
from typing import Dict, List, Union
import itertools

# Softwares

julia: str = "../julia-1.6.3/bin/julia"
plmdca: Path = Path("./src/plmdca.jl")
hhblits = (
    "../hhsuite/bin/hhblits"
)
uniclust = "../hhsuite/Database/UniRef30_2020_01"

# Softwares to build DCA models on Pfam domains

hmmpress = "hmmpress"
hmmscan = "hmmscan"
hmmalign: str = "hmmalign"

# Data paths

data_folder: Path = Path("./results")
tmp_folder: Path = Path("./tmp")
models_folder: Path = data_folder / "dca_models"
pfam_db_folder: Path = data_folder / "pfam" / "database"
pfam_files_folder: Path = data_folder / "pfam" / "files"
tree_folder: Path = data_folder / "phylogenies"
df_folder: Path = data_folder / "seq_ids"
aln_seq_folder: Path = data_folder / "aln_seq"
ancestral_seq_folder: Path = data_folder / "ancestral_sequences"
test_folder: Path = data_folder / "test"


def get_path(folder_type: str) -> Union[None, Path]:
    """Return path to folder.

    Parameters
    ----------
    folder_type : str
        Type of folder

    Returns
    -------
    pathlib.Path
        Path to folder (or None
        if folder_type does not
        exist)
    """
    folders = {
        "tmp": tmp_folder,
        "dca": models_folder,
        "tree": tree_folder,
        "seq_id": df_folder,
        "msa": aln_seq_folder,
        "ancestral": ancestral_seq_folder,
        "test": test_folder,
    }
    if folder_type in folders.keys():
        return folders[folder_type]
    return None


# Default values

random_seed: int = 10000
N_MSA: int = 200
prop_id: float = 0.9
prop_gap: float = 0.5
priority_coefs: Dict = {"pfam": 1, "hhblits": 2, "unknown": 3}

prop_gap_MSA: float = 0.3
N_poly: int = 30
N_CI: int = 100
bin_size: float = 0.5
itmd_outputs: bool = False




# Biological global constants

AAs: List = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "-",
]
nucleotides: List = ["A", "C", "G", "T", "-"]
codons: List = [
    f"{i}{j}{k}"
    for i, j, k in itertools.product(list("ACGT"), repeat=3)
]
translation_table: Dict = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGC": "C",
    "TGT": "C",
    "TGA": "*",
    "TGG": "W",
    "---": "-",
}
