import torch
from torch import nn

class MHASequenceShortener(nn.Module):
    def __init__(self, target_len, **ma_kwargs):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(**ma_kwargs)
        self.query = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, target_len, ma_kwargs['embed_dim']), 0., 0.2))

    def forward(self, x):
        x = self.multihead_attn(self.query.repeat(x.size(0),1,1), x, x,average_attn_weights=False) #this repeat method works for nested tensors and batched sequences
        x = (x[0] + self.query, x[1:])
        return x

# with layer norm
class MHASequenceShortenerWithLN(MHASequenceShortener):
    def __init__(self, target_len, **ma_kwargs):
        super().__init__(target_len, **ma_kwargs)
        self.layer_norm = nn.LayerNorm(ma_kwargs['kdim'])

    def forward(self, x):
        x = self.layer_norm(x)
        return super().forward(x)
    
# for GPT-like classifiers, cls_first can be set to False to append the cls token at the end of the shortened sequence.
class AdaptedModel(nn.Module):
    def __init__(self, embed_dim, seq_shortener, model, cls_first=True):
        super().__init__()
        self.seq_shortener = seq_shortener
        self.model = model
        self.cls = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, 1, embed_dim), 0., 0.2)) # bs=1, seq_len=1, embed_dim
        self.cls_first = cls_first
    
    def forward(self, x, output_attentions=True, output_hidden_states=True):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x, seq_shortener_attentions = self.seq_shortener(x) # second tuple element would be the attention heads
        if self.cls_first:
            x = torch.cat([self.cls.repeat(x.shape[0],1,1), x], dim=1)
        else:
            x = torch.cat([x, self.cls.repeat(x.shape[0],1,1)], dim=1)
        x = self.model(inputs_embeds=x, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if output_attentions:
            x['attentions'] = (seq_shortener_attentions, * x['attentions'])
        return x

def freeze_model(model):
    for name, param in model.named_parameters():
        if any([x in name for x in ['encoder','decoder','transformer']]):
            param.requires_grad = False
        if any([x in name for x in ['LayerNorm','layer_norm', '.ln_']]):
            param.requires_grad = True
    return model

"""Example:

import torch
from transformers import AutoModelForSequenceClassification

lm_classifier = freeze_model(AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2))
seq_shortener = MHASequenceShortenerWithLN(target_len=256, embed_dim=768, kdim=1280, vdim=1280, num_heads=4, batch_first=True)
adapted_lm = AdaptedModel(model=lm_classifier, seq_shortener=seq_shortener, embed_dim=768)

x = torch.rand([1,5000,1280])
y = adapted_lm(x)

"""
