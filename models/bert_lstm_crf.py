import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel
from utils import match_kwargs
from torchcrf import CRF

class NERNetwork(nn.Module):
    """A Generic Network for NERDA models.
    The network has an analogous architecture to the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).
    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, config, n_tags: int,using_lstm: bool = True, dropout: float = 0.1) -> None:
        """Initialize a NERDA Network
        Args:
            bert_model (nn.Module): huggingface `torch` transformers.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERNetwork, self).__init__()
        
        # extract AutoConfig, from which relevant parameters can be extracted.
        bert_model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.bert_encoder = AutoModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.using_lstm=using_lstm
        out_size=self.bert_encoder.config.hidden_size
        if self.using_lstm:
            self.lstm=nn.LSTM(self.bert_encoder.config.hidden_size,config.lstm_hidden_size,num_layers=1, bidirectional=True, batch_first=True)
            out_size=config.lstm_hidden_size*2
        
        self.hidden2tags = nn.Linear(out_size, n_tags)#BERT+Linear
        self.crf_layer=CRF(num_tags=n_tags,batch_first=True)

    def tag_outputs(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor,
                ) -> torch.Tensor:

        bert_model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
            }
        
        # match args with bert_model
        # bert_model_inputs = match_kwargs(self.bert_encoder.forward, **bert_model_inputs)
           
        outputs = self.bert_encoder(**bert_model_inputs)
        # apply drop-out
        last_hidden_state=outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)

        if self.using_lstm:
            last_hidden_state,_=self.lstm(last_hidden_state)
        # last_hidden_state for all labels/tags
        emissions = self.hidden2tags(last_hidden_state)

        return emissions
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor
                ):
        """Model Forward Iteration
        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention attention_mask.
            token_type_ids (torch.Tensor): Token Type IDs.
        Returns:
            torch.Tensor: predicted values.
        """

        emissions=self.tag_outputs(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        loss=-1*self.crf_layer(emissions=emissions,tags=target_tags,mask=attention_mask.byte())
        return loss
    
    def predict(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor,
                ):
        emissions=self.tag_outputs(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        return self.crf_layer.decode(emissions=emissions,mask=attention_mask.byte())