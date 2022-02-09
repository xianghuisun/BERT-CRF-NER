import torch
from torch._C import device
import torch.nn as nn
from transformers import AutoConfig,AutoModel
from transformers.utils.dummy_pt_objects import LogitsProcessor
from utils import match_kwargs
from torchcrf import CRF

class NERNetwork(nn.Module):
    """A Generic Network for NERDA models.
    The network has an analogous architecture to the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).
    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, config, n_tags: int, dropout: float = 0.1, device='cuda') -> None:
        """Initialize a NERDA Network
        Args:
            bert_model (nn.Module): huggingface `torch` transformers.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERNetwork, self).__init__()
        
        # extract AutoConfig, from which relevant parameters can be extracted.
        self.bert_model = AutoModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        out_size=self.bert_model.config.hidden_size
        self.lstm_layer=nn.LSTM(self.bert_model.config.hidden_size,config.lstm_hidden_size,num_layers=1, bidirectional=True, batch_first=True,dropout=0.5)
        out_size=config.lstm_hidden_size*2
        self.n_tags=n_tags
        self.device=device
        self.hidden2tags = nn.Linear(out_size, n_tags)#BERT+Linear

    def compute_loss(self, preds, target_tags, masks):
        
        # initialize loss function.
        lfn = torch.nn.CrossEntropyLoss()

        # Compute active loss to not compute loss of paddings
        active_loss = masks.view(-1) == 1

        active_logits = preds.view(-1, self.n_tags)
        active_labels = torch.where(
            active_loss,
            target_tags.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target_tags)
        )

        active_labels = torch.as_tensor(active_labels, device = torch.device(self.device), dtype = torch.long)
        
        # Only compute loss on actual token predictions
        loss = lfn(active_logits, active_labels)

        return loss

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor=None
                ):

        bert_model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
            }
        
        # match args with bert_model
        # bert_model_inputs = match_kwargs(self.bert_model.forward, **bert_model_inputs)
           
        outputs = self.bert_model(**bert_model_inputs)
        # apply drop-out
        last_hidden_state=outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state,_=self.lstm_layer(last_hidden_state)
        # last_hidden_state for all labels/tags
        logits = self.hidden2tags(last_hidden_state)
        if target_tags!=None:
            loss=self.compute_loss(preds=logits,target_tags=target_tags,masks=attention_mask)
            return loss
        else:
            return logits