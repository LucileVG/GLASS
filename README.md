# Investigating recent events of selection in _E. coli_ with Gene-Level Amino-acid Score Shift (GLASS)

GLASS (Gene-Level Amino-acid Score Shift) is a new selection test based on the predicted effects of non-synonymous mutations. It compares the distribution of effects of mutations observed on a gene to the distribution expected in the absence of selection in order to quantify the intensity of purifying selection acting on a gene. Here, we use Direct-Coupling Analysis ([DCA](https://en.wikipedia.org/wiki/Direct_coupling_analysis)) to predict the neutral, deleterious or beneficial effect of non-synonymous mutations.

Paper: [Predicting the effect of mutations to investigate recent events of selection across 60,472 _Escherichia coli_ strains](link to the paper) (Vigué L.and Tenaillon O.)

## Installation
The following softwares/libraries are required to run the code:
- python3: the code was tested on python v3.9, with following libraries:
    - standard libraries:
        - pathlib
        - argparse
        - typing
        - itertools
        - random
    - non-standard libraries:
        - joblib 1.1.0
        - biopython 1.79
        - pandas 1.4.4
        - numpy 1.19.5
        - pot 0.8.1.0
- [julia](https://julialang.org/): to train DCA models (tested on julia v1.6.3) with the following packages installed: plmDCA (https://github.com/pagnani/PlmDCA), NPZ and DCAUtils.
- [mafft](https://mafft.cbrc.jp/alignment/software/): to align sequences (tested on v7.407)
- [fasttree](http://www.microbesonline.org/fasttree/): to build phylogenies (tested on v2.1.10)
- [iqtree](http://www.iqtree.org/): to infer ancestral sequences (tested on v.2.0.3)
- [hhblits](https://github.com/soedinglab/hh-suite): to find distant homologs of a protein for training DCA models (tested on v.3.3.0 with UniRef30 2020-01 database)

## Paths

Open the file  ```src/config.py``` and modify paths in the software section according to your software configuration.


## Demo

Run the following commands to test the demo:

```
python3 main_hhblits.py  --input demo_data/consensus.fasta  --nthreads 3
python3 process_sequences.py --consensus demo_data/consensus.fasta --outgroup demo_data/Salmonella_enterica_FDAARGOS_609-WGS_VFAF01.1.fasta --outgroup_hits demo_data/hits.csv --sequences_dir demo_data/sequences --nthreads 3
```
