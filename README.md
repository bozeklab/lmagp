# lmagp
Repository of the paper "Language models are good pathologists: using attention-based sequence reduction and text-pretrained transformers for efficient WSI classification"

## Set up
Required packages:
- The models' implementation only require `pytorch`. We use language models from the `transformers` library.
- The packages required for training are `pytorch-lightning`, `hydra-zen`, `torchmetrics`.
- `conda env create -f env.yaml` will create a conda environment `lmagp-env` with the required dependencies

## Methods
The methods proposed in the paper are available in `src/models/adapted_transformers.py` . A [_frozen pre-trained_](https://arxiv.org/abs/2103.05247) RoBERTa-base with a SeqShort layer can be instantiated in the following way:

```python
import torch
from transformers import AutoModelForSequenceClassification
from src.models.adapted_transformers import AdaptedModel, MHASequenceShortenerWithLN, freeze_model

lm_classifier = freeze_model(AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)) # will freeze the encoder parameters except for the layer norm layers.
seq_shortener = MHASequenceShortenerWithLN(target_len=256, embed_dim=768, kdim=1280, vdim=1280, num_heads=4, batch_first=True) # kdim, vdim: hidden dim of efficientnet v2 l
adapted_lm = AdaptedModel(model=lm_classifier, seq_shortener=seq_shortener, embed_dim=768)

x = torch.rand([1,5000,1280]) # [batch_size, num_tiles, feature_extractor_hidden_dim]
y = adapted_lm(x) # y['attentions'][0] will have the attention matrix of the SeqShort layer
```
The models should be easily plugged to an existing Multiple Instance Learning pipeline.

## Training:
### Data
We require the WSIs to be preprocessed previously. Our implementation relies on each sample being a `dict` (saved as a .pickle file) with a `features` key, whose value is a `torch.Tensor` of shape `[num_tiles, feature_extractor_hidden_dim]`. For example for a WSI comprising 100 tiles whose features were obtained with EfficientNetV2-L, the `features` field should be a tensor of shape `[100, 1280]`.

The `csvs` directory contain the TCGA-BRCA splits that were used in our study. 

### Running
We use [Hydra]() and [PyTorch-Lightning]() for training, and every hyperparameter is configurable from .yaml config files. We provide an example config, where it is only needed to specify the correct path to the 10-fold .csvs root, and the fold number for which we want to train, validate and test.
```
# conda env create -f env.yaml
conda activate lmagp-env
python3 train_classifier.py -cp configs -cn seq-short-roberta-base.yaml ++csvs_root=/path/to/the/csvs ++fold=0
```

### Making heatmaps:
Once the model was trained, it is possible to make attention rollout heatmaps for the test set of the configuration file:
```
python3 make_attention_rollout.py -cp configs -cn seq-short-roberta-base.yaml \
    ++hydra.run.dir=outputs/attention_heatmaps/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    ++checkpoint=path/to/checkpoint.ckpt
```
