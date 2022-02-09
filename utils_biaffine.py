import torch
from torch import nn
import torch
import logging
logger=logging.getLogger('bilstm-biaffine.span_loss')

class Span_loss(nn.Module):
    def __init__(self, num_label, class_weight=None):
        super().__init__()
        self.num_label = num_label
        if class_weight != None:
            self.class_weight=torch.FloatTensor(class_weight)
            logger.info("Class weight : {}".format(self.class_weight.cpu().tolist()))
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)#reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()#reduction='mean'

    def forward(self,span_logits,span_label,span_mask):
        '''
        span_logits.size()==(bsz,max_length,max_length,num_labels)
        span_label.size()==span_mask.size()==(bsz,max_length,max_length)
        span_label只有(i,j)位置是1,(i,j)代表句子的第i个位置与第j个位置之间的span是一个ent
        span_mask左上半部分是1
        '''
        #print(span_mask.size(),'spanmask.size()')
        mask_pad=span_mask.view(-1)==1
        span_label = span_label.view(size=(-1,))[mask_pad]#(bsz*max_length*max_length,)
        span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]#(bsz*max_length*max_length,num_labels)
        span_loss = self.loss_func(input=span_logits, target=span_label)#(bsz*max_length*max_length,)
        
        # print("span_logits : ",span_logits.size())
        # print("span_label : ",span_label.size())
        # print("span_mask : ",span_mask.size())
        # print("span_loss : ",span_loss.size())

        # start_extend = span_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = span_mask.unsqueeze(1).expand(-1, seq_len, -1)
        # span_mask = span_mask.view(size=(-1,))#(bsz*max_length*max_length,)
        # span_loss *=span_mask
        #avg_se_loss = torch.sum(span_loss) / span_mask.size()[0]
        # avg_se_loss = torch.sum(span_loss) / torch.sum(span_mask).item()
        # # avg_se_loss = torch.sum(sum_loss) / bsz
        # return avg_se_loss
        return span_loss

class metrics_span(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, span_mask):
        '''
        logits.size()==(bsz,max_length,max_length,num_labels) .score for each span
        labels.size()==(bsz,max_length,max_length)
        span_mask.size()==(bsz,max_length,max_length)#下三角全1是，上三角超出句子实际长度的位置也是1
        '''
        (bsz,max_length,max_length,num_labels)=logits.size()
        assert labels.size()==(bsz,max_length,max_length)==span_mask.size()
        #print(bsz,max_length,max_length,num_labels)

        span_mask.unsqueeze_(-1)#(bsz,max_length,max_length,1)
        assert span_mask.size()==(bsz,max_length,max_length,1)
        logits*=span_mask
        logits = torch.argmax(logits,dim=-1)#(bsz,max_length,max_length)
        assert logits.size()==(bsz,max_length,max_length)#label_id for each span

        logits=logits.view(size=(-1,)).float()
        labels=labels.view(size=(-1,)).float()

        ones=torch.ones_like(logits)
        zero=torch.zeros_like(logits)
        y_pred=torch.where(logits<1,zero,ones)
        y_pred=torch.triu(y_pred)#extract upper triangle matrix

        ones=torch.ones_like(labels)
        zero=torch.zeros_like(labels)
        y_true=torch.where(labels<1,zero,ones)#only golden span position is 1, otherwise positions are zeros

        corr=torch.eq(logits,labels).float()
        corr=torch.mul(corr,y_true)
        
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        
        f1=2*recall*precision/(recall+precision+1e-8)

        if torch.sum(labels)==0 and torch.sum(logits)==0:
            return 1.0,1.0,1.0

        return recall.item(), precision.item(), f1.item()


