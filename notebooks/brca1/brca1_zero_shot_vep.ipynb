{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "17440de8",
   "metadata": {},
   "source": [
    "## Zero-shot prediction of *BRCA1* variant effects with Evo 2\n",
    "\n",
    "The human *BRCA1* gene encodes for a protein that repairs damaged DNA ([Moynahan et al., 1999](https://www.cell.com/molecular-cell/fulltext/S1097-2765%2800%2980202-6)). Certain variants of this gene have been associated with an increased risk of breast and ovarian cancers ([Miki et al., 1994](https://www.science.org/doi/10.1126/science.7545954?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)). Using Evo 2, we can predict whether a particular single nucleotide variant (SNV) of the *BRCA1* gene is likely to be harmful to the protein's function, and thus potentially increase the risk of cancer for the patient with the genetic variant.\n",
    "\n",
    "We start by loading a dataset from [Findlay et al. (2018)](https://www.nature.com/articles/s41586-018-0461-z), which contains experimentally measured function scores of 3,893 *BRCA1* SNVs. These function scores reflect the extent by which the genetic variant has disrupted the protein's function, with lower scores indicating greater disruption. In this dataset, the SNVs are classified into three categories based on their function scores: `LOF` (loss-of-function), `INT` (intermediate), and `FUNC` (functional). We start by reading in this dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f090aadb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/bin/bash: /home/ggsun/miniconda/envs/evo2-release/lib/libtinfo.so.6: no version information available (required by /bin/bash)\n",
      "Requirement already satisfied: matplotlib in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (3.10.0)\n",
      "Requirement already satisfied: pandas in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (2.2.3)\n",
      "Requirement already satisfied: seaborn in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (0.13.2)\n",
      "Requirement already satisfied: scikit-learn in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (1.6.1)\n",
      "Requirement already satisfied: openpyxl in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (3.1.5)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (1.3.1)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (0.12.1)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (4.56.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (1.4.8)\n",
      "Requirement already satisfied: numpy>=1.23 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (2.2.3)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (24.2)\n",
      "Requirement already satisfied: pillow>=8 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (11.1.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (3.2.1)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from matplotlib) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from pandas) (2025.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from pandas) (2025.1)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from scikit-learn) (1.15.2)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from scikit-learn) (3.5.0)\n",
      "Requirement already satisfied: et-xmlfile in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from openpyxl) (2.0.0)\n",
      "Requirement already satisfied: six>=1.5 in /home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "# Install dependencies\n",
    "!pip install matplotlib pandas seaborn scikit-learn openpyxl\n",
    "\n",
    "# Required imports\n",
    "from Bio import SeqIO\n",
    "import gzip\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "# Set root path\n",
    "os.chdir('../..')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1e26f1bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>chromosome</th>\n",
       "      <th>position (hg19)</th>\n",
       "      <th>reference</th>\n",
       "      <th>alt</th>\n",
       "      <th>function.score.mean</th>\n",
       "      <th>func.class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.372611</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.045313</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.108254</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.277963</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.388414</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.280973</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.973683</td>\n",
       "      <td>INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.373489</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>0.006314</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>17</td>\n",
       "      <td>41276132</td>\n",
       "      <td>A</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.207552</td>\n",
       "      <td>FUNC</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   chromosome  position (hg19) reference alt  function.score.mean func.class\n",
       "0          17         41276135         T   G            -0.372611       FUNC\n",
       "1          17         41276135         T   C            -0.045313       FUNC\n",
       "2          17         41276135         T   A            -0.108254       FUNC\n",
       "3          17         41276134         T   G            -0.277963       FUNC\n",
       "4          17         41276134         T   C            -0.388414       FUNC\n",
       "5          17         41276134         T   A            -0.280973       FUNC\n",
       "6          17         41276133         C   T            -0.973683        INT\n",
       "7          17         41276133         C   G            -0.373489       FUNC\n",
       "8          17         41276133         C   A             0.006314       FUNC\n",
       "9          17         41276132         A   T            -0.207552       FUNC"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brca1_df = pd.read_excel(\n",
    "    os.path.join('notebooks', 'brca1', '41586_2018_461_MOESM3_ESM.xlsx'),\n",
    "    header=2,\n",
    ")\n",
    "brca1_df = brca1_df[[\n",
    "    'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',\n",
    "]]\n",
    "\n",
    "brca1_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13e0c7d5",
   "metadata": {},
   "source": [
    "We then group the `FUNC` and `INT` classes of SNVs together into a single category (`FUNC/INT`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ce7df7cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>chrom</th>\n",
       "      <th>pos</th>\n",
       "      <th>ref</th>\n",
       "      <th>alt</th>\n",
       "      <th>score</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.372611</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.045313</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.108254</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.277963</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.388414</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.280973</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.973683</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.373489</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>0.006314</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>17</td>\n",
       "      <td>41276132</td>\n",
       "      <td>A</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.207552</td>\n",
       "      <td>FUNC/INT</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   chrom       pos ref alt     score     class\n",
       "0     17  41276135   T   G -0.372611  FUNC/INT\n",
       "1     17  41276135   T   C -0.045313  FUNC/INT\n",
       "2     17  41276135   T   A -0.108254  FUNC/INT\n",
       "3     17  41276134   T   G -0.277963  FUNC/INT\n",
       "4     17  41276134   T   C -0.388414  FUNC/INT\n",
       "5     17  41276134   T   A -0.280973  FUNC/INT\n",
       "6     17  41276133   C   T -0.973683  FUNC/INT\n",
       "7     17  41276133   C   G -0.373489  FUNC/INT\n",
       "8     17  41276133   C   A  0.006314  FUNC/INT\n",
       "9     17  41276132   A   T -0.207552  FUNC/INT"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Rename columns\n",
    "brca1_df.rename(columns={\n",
    "    'chromosome': 'chrom',\n",
    "    'position (hg19)': 'pos',\n",
    "    'reference': 'ref',\n",
    "    'alt': 'alt',\n",
    "    'function.score.mean': 'score',\n",
    "    'func.class': 'class',\n",
    "}, inplace=True)\n",
    "\n",
    "# Convert to two-class system\n",
    "brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')\n",
    "\n",
    "brca1_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ef6f10c",
   "metadata": {},
   "source": [
    "We build a function to parse the reference and variant sequences of a 8,192-bp window around the genomic position of each SNV, using the reference sequence of human chromosome 17 where *BRCA1* is located."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4be1bb8e",
   "metadata": {
    "lines_to_next_cell": 2,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "chrom          17\n",
      "pos      41276135\n",
      "ref             T\n",
      "alt             G\n",
      "score   -0.372611\n",
      "class    FUNC/INT\n",
      "Name: 0, dtype: object\n",
      "--\n",
      "Reference, SNV 0: ...TGTTCCAATGAACTTTAACACATTAGAAAA...\n",
      "Variant, SNV 0:   ...TGTTCCAATGAACTGTAACACATTAGAAAA...\n"
     ]
    }
   ],
   "source": [
    "WINDOW_SIZE = 8192\n",
    "\n",
    "# Read the reference genome sequence of chromosome 17\n",
    "with gzip.open(os.path.join('notebooks', 'brca1', 'GRCh37.p13_chr17.fna.gz'), \"rt\") as handle:\n",
    "    for record in SeqIO.parse(handle, \"fasta\"):\n",
    "        seq_chr17 = str(record.seq)\n",
    "        break\n",
    "\n",
    "def parse_sequences(pos, ref, alt):\n",
    "    \"\"\"\n",
    "    Parse reference and variant sequences from the reference genome sequence.\n",
    "    \"\"\"\n",
    "    p = pos - 1 # Convert to 0-indexed position\n",
    "    full_seq = seq_chr17\n",
    "\n",
    "    ref_seq_start = max(0, p - WINDOW_SIZE//2)\n",
    "    ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)\n",
    "    ref_seq = seq_chr17[ref_seq_start:ref_seq_end]\n",
    "    snv_pos_in_ref = min(WINDOW_SIZE//2, p)\n",
    "    var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref+1:]\n",
    "\n",
    "    # Sanity checks\n",
    "    assert len(var_seq) == len(ref_seq)\n",
    "    assert ref_seq[snv_pos_in_ref] == ref\n",
    "    assert var_seq[snv_pos_in_ref] == alt\n",
    "\n",
    "    return ref_seq, var_seq\n",
    "\n",
    "# Parse sequences for the first variant\n",
    "row = brca1_df.iloc[0]\n",
    "ref_seq, var_seq = parse_sequences(row['pos'], row['ref'], row['alt'])\n",
    "\n",
    "print(row)\n",
    "print('--')\n",
    "print(f'Reference, SNV 0: ...{ref_seq[4082:4112]}...')\n",
    "print(f'Variant, SNV 0:   ...{var_seq[4082:4112]}...')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5acd3e9a-0b33-44c4-a95a-9be49ef61a76",
   "metadata": {},
   "source": [
    "Then, we load Evo 2 1B and score the likelihoods of the reference and variant sequences of each SNV. (Note: we use the smaller Evo 2 1B base model here as a quick demonstration, but we strongly recommend using the larger Evo 2 7B and 40B models for more accurate predictions.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "362d5a24",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tokenizers not found, unable to use HFAutoTokenizer\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:00<00:00, 220.42it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extra keys in state_dict: {'blocks.3.mixer.dense._extra_state', 'blocks.24.mixer.dense._extra_state', 'unembed.weight', 'blocks.10.mixer.dense._extra_state', 'blocks.17.mixer.dense._extra_state'}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ggsun/miniconda/envs/evo2-release/lib/python3.12/site-packages/transformer_engine/pytorch/module/base.py:630: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.\n",
      "  state = torch.load(state, map_location=\"cuda\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loaded model evo2_1b_base from /home/ggsun/.cache/huggingface/hub/models--arcinstitute--evo2_1b_base/snapshots/6915b21845659a78b55e59a1eb603039fc81c49f/evo2_1b_base.pt!\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ggsun/evo2/vortex/vortex/model/utils.py:153: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.\n",
      "  return torch_load(state, map_location=device)\n"
     ]
    }
   ],
   "source": [
    "from evo2.models import Evo2\n",
    "\n",
    "# Load model\n",
    "model = Evo2('evo2_1b_base')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "135bffe8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Scoring likelihoods of 1326 reference sequences with Evo 2...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1326/1326 [02:20<00:00,  9.45it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Scoring likelihoods of 3893 variant sequences with Evo 2...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████████████████| 3893/3893 [06:49<00:00,  9.51it/s]\n"
     ]
    }
   ],
   "source": [
    "# Build mappings of unique reference sequences\n",
    "ref_seqs = []\n",
    "ref_seq_to_index = {}\n",
    "\n",
    "# Parse sequences and store indexes\n",
    "ref_seq_indexes = []\n",
    "var_seqs = []\n",
    "\n",
    "for _, row in brca1_df.iterrows():\n",
    "    ref_seq, var_seq = parse_sequences(row['pos'], row['ref'], row['alt'])\n",
    "\n",
    "    # Get or create index for reference sequence\n",
    "    if ref_seq not in ref_seq_to_index:\n",
    "        ref_seq_to_index[ref_seq] = len(ref_seqs)\n",
    "        ref_seqs.append(ref_seq)\n",
    "    \n",
    "    ref_seq_indexes.append(ref_seq_to_index[ref_seq])\n",
    "    var_seqs.append(var_seq)\n",
    "\n",
    "ref_seq_indexes = np.array(ref_seq_indexes)\n",
    "\n",
    "print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')\n",
    "ref_scores = model.score_sequences(ref_seqs)\n",
    "\n",
    "print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')\n",
    "var_scores = model.score_sequences(var_seqs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbf2de1e-17b1-4f0d-9004-1eb917ed83ac",
   "metadata": {},
   "source": [
    "We calculate the change in likelihoods for each variant relative to the likelihood of their respective wild-type sequence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a49d5859-87ed-49c9-8e3d-18629f073022",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>chrom</th>\n",
       "      <th>pos</th>\n",
       "      <th>ref</th>\n",
       "      <th>alt</th>\n",
       "      <th>score</th>\n",
       "      <th>class</th>\n",
       "      <th>evo2_delta_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.372611</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>-0.000054</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.045313</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000140</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>17</td>\n",
       "      <td>41276135</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.108254</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000074</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.277963</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>-0.000146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>C</td>\n",
       "      <td>-0.388414</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>-0.000104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>17</td>\n",
       "      <td>41276134</td>\n",
       "      <td>T</td>\n",
       "      <td>A</td>\n",
       "      <td>-0.280973</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.973683</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000299</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>G</td>\n",
       "      <td>-0.373489</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000077</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>17</td>\n",
       "      <td>41276133</td>\n",
       "      <td>C</td>\n",
       "      <td>A</td>\n",
       "      <td>0.006314</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>0.000352</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>17</td>\n",
       "      <td>41276132</td>\n",
       "      <td>A</td>\n",
       "      <td>T</td>\n",
       "      <td>-0.207552</td>\n",
       "      <td>FUNC/INT</td>\n",
       "      <td>-0.000300</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   chrom       pos ref alt     score     class  evo2_delta_score\n",
       "0     17  41276135   T   G -0.372611  FUNC/INT         -0.000054\n",
       "1     17  41276135   T   C -0.045313  FUNC/INT          0.000140\n",
       "2     17  41276135   T   A -0.108254  FUNC/INT          0.000074\n",
       "3     17  41276134   T   G -0.277963  FUNC/INT         -0.000146\n",
       "4     17  41276134   T   C -0.388414  FUNC/INT         -0.000104\n",
       "5     17  41276134   T   A -0.280973  FUNC/INT          0.000084\n",
       "6     17  41276133   C   T -0.973683  FUNC/INT          0.000299\n",
       "7     17  41276133   C   G -0.373489  FUNC/INT          0.000077\n",
       "8     17  41276133   C   A  0.006314  FUNC/INT          0.000352\n",
       "9     17  41276132   A   T -0.207552  FUNC/INT         -0.000300"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Subtract score of corresponding reference sequences from scores of variant sequences\n",
    "delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]\n",
    "\n",
    "# Add delta scores to dataframe\n",
    "brca1_df[f'evo2_delta_score'] = delta_scores\n",
    "\n",
    "brca1_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20aea762-ef7f-4687-88ca-56d9042e7a0d",
   "metadata": {},
   "source": [
    "This delta likelihood should be predictive of how disruptive the SNV is to the protein's function: the lower the delta, the more likely that the SNV is disruptive. We can show this by comparing the distributions of delta likelihoods for the two classes of SNVs (functional/intermediate vs loss-of-function)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0c27729e-927e-42ec-b311-1e3d901eb29e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(4, 2))\n",
    "\n",
    "# Plot stripplot of distributions\n",
    "p = sns.stripplot(\n",
    "    data=brca1_df,\n",
    "    x='evo2_delta_score',\n",
    "    y='class',\n",
    "    hue='class',\n",
    "    order=['FUNC/INT', 'LOF'],\n",
    "    palette=['#777777', 'C3'],\n",
    "    size=2,\n",
    "    jitter=0.3,\n",
    ")\n",
    "\n",
    "# Mark medians from each distribution\n",
    "sns.boxplot(showmeans=True,\n",
    "            meanline=True,\n",
    "            meanprops={'visible': False},\n",
    "            medianprops={'color': 'k', 'ls': '-', 'lw': 2},\n",
    "            whiskerprops={'visible': False},\n",
    "            zorder=10,\n",
    "            x=\"evo2_delta_score\",\n",
    "            y=\"class\",\n",
    "            data=brca1_df,\n",
    "            showfliers=False,\n",
    "            showbox=False,\n",
    "            showcaps=False,\n",
    "            ax=p)\n",
    "plt.xlabel('Delta likelihood score, Evo 2')\n",
    "plt.ylabel('BRCA1 SNV class')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3974e39-6c50-4503-9bab-829b1ac1b14a",
   "metadata": {},
   "source": [
    "We can also calculate the area under the receiver operating characteristic curve (AUROC) of this zero-shot prediction method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f9e6cc5e-9c98-4010-8210-b38f570e1290",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Zero-shot prediction AUROC: 0.73\n"
     ]
    }
   ],
   "source": [
    "# Calculate AUROC of zero-shot predictions\n",
    "y_true = (brca1_df['class'] == 'LOF')\n",
    "auroc = roc_auc_score(y_true, -brca1_df['evo2_delta_score'])\n",
    "\n",
    "print(f'Zero-shot prediction AUROC: {auroc:.2}')"
   ]
  }
 ],
 "metadata": {
  "jupytext": {
   "cell_metadata_filter": "-all",
   "main_language": "python",
   "notebook_metadata_filter": "-all"
  },
  "kernelspec": {
   "display_name": "evo2-release",
   "language": "python",
   "name": "evo2-release"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
