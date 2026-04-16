# TO DO make a nice README

# Mention the need for 3 different env !!! and explain which is what !!! 

# Explain the whole ordeal of ESM2 and only running it on C.Elegan / Drosophila / E.Coli 

## Environment installation SeqVec:

This is because SeqVec comes from Bio-embeddings which is an old package that is not longer maintained and has very specific requirements. 

```bash

# Or conda
mamba create -n seq_vec_emb --file environment/seq_vec_emb_detailed_env.txt

mamba activate seq_vec_emb

python -m pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.0+cu113

python -m pip install "bio-embeddings[seqvec] @ git+https://github.com/sacdallago/bio_embeddings.git@develop"

```

## Environment installation ESM2:

```bash

# Or conda
mamba create -n esm_emb python=3.10 pip -y

mamba activate esm_emb

pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install transformers accelerate
```

## Changes to the orignal code of Graph-BERT: 

GPU selection + CPU limitation was added, these were changes in files script_1 / 2 / 3 and 4 in both the `Graph-BERT` and `Modiefied_Graph-BERT` folders. These changes are clearly specified in each of the files via a docstring that explains each individual change. Overall, the behavior of the code was not altered just adapted to suit a shared laboratory cluster.  

## Dataset and SeqVec / ESM-2 weights:

Concerning the original and produced datasets, and especially ESM-2 weights these are all present on this Google drive [https://drive.google.com/drive/folders/1ebuwW51Fo05qFbxVlUjVwS31oRxY5D06?usp=sharing]. This is due to GitHub's size limit for pushes and individual files. The specified google drive possess five different folders: 

1. `data` which contains the original data used by the authors, this same folder is also present in the `Graph-BERT` folder on this repository
2. `esm-files`, which contains the ESM-2 dictionaries for the used datasets in the 650M and 3B parameter settings
3. `Node_creation`, which contains the node files created for this project
4. `S-VGAE` data which is the original data from which the authors have created their own node and link files, this was used in this project for the re-creation of the node files
5. `seqvec_files`, which contain the SeqVec weight dictionaries that were produced for this project 

