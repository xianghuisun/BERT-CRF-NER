import torch
import torch.nn as nn
from transformers import AutoModel

class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        '''
        (bsz,max_length,dim) x.size()==y.size()
        '''
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        batch_size,seq_len,hidden=x.shape
        x=x.view(-1,hidden)
        bilinar_mapping=torch.matmul(x,self.U.view(hidden,-1))#(batch_size*seq_len,out_size*hidden)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)#(batch_size,hidden_size,seq_len)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)#(batch_size,seq_len*out_size,seq_len)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        
        #bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        #(bsz,max_length,max_length,num_labels)
        return bilinar_mapping

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
        self.lstm_layer=nn.LSTM(self.bert_model.config.hidden_size,config.lstm_hidden_size,num_layers=config.lstm_layer, bidirectional=True, batch_first=True,dropout=0.5)
        self.n_tags=n_tags
        self.device=device
        #将LSTM的输出输入到两个Linear中
        self.ner_start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*config.lstm_hidden_size, out_features=config.to_biaffine_size),
                                            torch.nn.ReLU(),nn.Dropout(config.project_dropout))
        self.ner_end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*config.lstm_hidden_size, out_features=config.to_biaffine_size),
                                            torch.nn.ReLU(),nn.Dropout(config.project_dropout))      
        
        self.ner_biaffine_layer=biaffine(in_size=config.to_biaffine_size,out_size=self.n_tags)#n_tags是包含O的，O代表non-span

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
        ###################################################BERT####################################
        outputs = self.bert_model(**bert_model_inputs)
        last_hidden_state=outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        batch_size,max_length,_=last_hidden_state.size()
        ###################################################BiLSTM####################################
        lstm_out,_=self.lstm_layer(last_hidden_state)
        ###################################################project layers###########################
        ner_start_rep = self.ner_start_layer(lstm_out) 
        ner_end_rep = self.ner_end_layer(lstm_out) 
        #(bsz,max_length,config.to_biaffine_size)
        ##################################################biaffine##################################
        ner_span_logits = self.ner_biaffine_layer(ner_start_rep, ner_end_rep)
        ner_span_logits = ner_span_logits.contiguous()
        assert ner_span_logits.size()==(batch_size,max_length,max_length,self.n_tags)
        return ner_span_logits#(bsz,max_length,max_length,num_labels)