# Evo 2: DNA modeling and design across all life's domains

</div>

Evo 2 is a state of the art DNA language model for long context modeling and design. Evo 2 uses the [StripedHyena 2](https://github.com/Zymrael/vortex) architecture and is pretrained using [Savanna](https://github.com/Zymrael/savanna) on 2048 GPUs. Evo 2 models DNA sequences at single-nucleotide resolution at up to 1 million base pair context length. Evo 2 is trained autoregressively on [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2), a dataset containing 8.8 trillion tokens from all domains of life.

We describe Evo 2 in the preprint:
["Genome modeling and design across all domains of life with Evo 2"]().

## Contents

- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training Code](#dataset)
- [Citation](#citation)


## Setup

### Requirements

Evo 2 is based on [StripedHyena 2](https://github.com/Zymrael/vortex). A CUDA-capable system is required to build and install the prerequisites. Evo 2 uses [FlashAttention](https://github.com/Dao-AILab/flash-attention), which may not work on all GPU architectures. Please consult the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention#installation-and-features) for the current list of supported GPUs. 

### Installation

Follow the commands below to install.

```bash
conda create -n evo2 python=3.12 -y && conda activate evo2
git clone https://github.com/arcinstitute/evo2.git
cd evo2/
git submodule init vortex
git submodule update vortex
pip install .
```

After installation, check that the installation was correct by running a test
```
python ./test/test_evo2.py --model_name evo2_7b
```

## Checkpoints

We provide the following model checkpoints, hosted on [HuggingFace](https://huggingface.co/arcinstitute):
| Checkpoint Name                        | Description |
|----------------------------------------|-------------|
| `evo2_40b`  | A model pretrained with 1 million context obtained through context extension of `evo2_40b_base`.|
| `evo2_7b`  | A model pretrained with 1 million context obtained through context extension of `evo2_7b_base`.|
| `evo2_40b_base`  | A model pretrained with 8192 context length.|
| `evo2_7b_base`  | A model pretrained with 8192 context length.|
| `evo2_1b_base`  | A smaller model pretrained with 8192 context length.|

To use Evo 2 40B, you will need multiple GPUs. Vortex automatically handles device placement, splitting the model across available cuda devices.

## Usage

Below are simple examples of how to download Evo 2 and use it locally using Python.

### Inference

Evo 2 can be used to score the likelihood of DNA sequence.

```python
import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')
evo2_model.model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

outputs, _ = evo2_model(input_ids)
logits = outputs[0]

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
```

### Generation

Evo 2 can generate DNA sequence based on prompts.

```python
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')
evo2_model.model.eval()

output = evo2_model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)
print(output.sequences[0])
```

### Embeddings

Evo 2 embeddings can be saved for use downstream.

```python
import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')
evo2_model.model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

layer_name = 'blocks.28.mlp.l3'

outputs, embeddings = evo2_model.forward(input_ids, return_embeddings=True, layer_names=[layer_name])

print('Embeddings shape: ', embeddings[layer_name].shape)
```

### Notebooks

- `notebooks/brca1/brca1_zero_shot_vep.ipynb`: Zero-shot *BRCA1* variant effect prediction with Evo 2.

## Dataset

The OpenGenome2 dataset used for pretraining Evo2 is available at [Hugging Face datasets](https://huggingface.co/datasets/LongSafari/open-genome). Data is available either as raw fastas or as JSONL files which include preprocessing and data augmentation.

## Training Code

Evo 2 was trained using [Savanna](https://github.com/Zymrael/savanna), an open source framework for training alternative architectures.

## Citation

If you find these models useful for your research, please cite the relevant papers

TODO
