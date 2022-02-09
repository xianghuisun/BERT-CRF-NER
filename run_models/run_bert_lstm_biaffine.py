import enum
import pandas as pd
import numpy as np
import torch
import argparse
import os,json

from torch.utils.data import dataloader
os.environ['TOKENIZERS_PARALLELISM']='true'

import sys
from tqdm import tqdm
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random

from preprocess_biaffine import yield_data,generate_label2id,process_nerlabel
from utils import compute_loss, get_ent_tags, batch_to_device, compute_f1
from models.bert_lstm_biaffine import NERNetwork
from utils_biaffine import Span_loss, metrics_span
from visdom import Visdom
import logging

save_dir='/home/xhsun/Desktop/NER_Parsing/train_models/biaffine'
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)

logger=logging.getLogger('bilstm-biaffine')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log/bert_lstm_biaffine_log.txt',mode='w')
fh.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# System based
random.seed(seed)
np.random.seed(seed)

vis=Visdom(env='main',log_to_filename='log/bert_lstm_biaffine_vis_log')

device='cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device {}".format(device))

def train(args,
        label2id,
        train_dataloader,
        test_dataloader,
        load_finetune=False):
    
    ner_num_label=len(label2id)
    
    logger.info("label2id : {}".format(label2id))
    print_loss_step=len(train_dataloader)//5
    evaluation_steps=len(train_dataloader)//2
    logger.info("Under an epoch, loss will be output every {} step, and the model will be evaluated every {} step".format(print_loss_step,evaluation_steps))
    model = NERNetwork(config=args,n_tags=ner_num_label)

    if load_finetune and os.path.exists(args.ckpt):
        logger.info("Loading fine-tuned model from {}".format(args.checkpoints))
        key_match_result=model.load_state_dict(torch.load(f=args.checkpoints,map_location='cpu'))
        logger.info(key_match_result)
    model.to(device)

    optimizer_parameters=model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = args.learning_rate)
    logger.info('length of dataloader : {}'.format(len(train_dataloader)))
    num_train_steps=len(train_dataloader)*args.epochs
    warmup_steps=int(num_train_steps*args.warmup_proportion)
    logger.info("num_train_steps : {}, warmup_proportion : {}, warmup_steps : {}".format(num_train_steps,args.warmup_proportion,warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )
    
    ner_span_loss_func = Span_loss(ner_num_label,class_weight=[1]+[8]*(ner_num_label-1)).to(device)
    span_acc=metrics_span().to(device)
    
    global_step=0
    best=-1
    # predictions=predict(model=model,test_dataloader=test_dataloader,tag_encoder=tag_encoder,device=device)
    # f1=compute_f1(pred_tags=predictions,golden_tags=test_conll_tags)
    # if f1>previous_f1:
    #     logger.info("Previous f1 score is {} and current f1 score is {}".format(previous_f1,f1))
    #     previous_f1=f1

    for epoch in range(args.epochs):
        model.train()
        model.zero_grad()
        training_loss=0.0
        for iteration, batch in tqdm(enumerate(train_dataloader)):
            batch=batch_to_device(inputs=batch,device=device)
            input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
            ner_span_label,ner_span_mask=batch['span_label'],batch['span_mask']

            ner_span_logits=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)#(batch_size,seq_length,num_classes)
            loss=ner_span_loss_func(span_logits=ner_span_logits,span_label=ner_span_label,span_mask=ner_span_mask).float().mean()
            #target_tags将CLS和SEP赋予标签O
            training_loss+=loss.item()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step+=1

            # if iteration % print_loss_step == 0:
            #     training_loss/=print_loss_step
            #     logger.info("Epoch : {}, global_step : {}/{}, loss_value : {} ".format(epoch,global_step,num_train_steps,training_loss))
            #     training_loss=0.0

            vis.line([optimizer.param_groups[0]['lr']],[global_step],win=f'learning rate',
                    opts=dict(title=f'learning rate', xlabel='step', ylabel='lr'), update='append')
            vis.line([loss.item()],[global_step],win=f'training loss',
                    opts=dict(title=f'training loss', xlabel='step', ylabel='loss'), update='append')

            if (iteration+1) % evaluation_steps == 0:
                # tar_ner_span_logits = torch.nn.functional.softmax(tar_ner_span_logits, dim=-1)
                # recall,precise,span_f1=span_acc(tar_ner_span_logits,tar_ner_span_label.to(device))
                # logger.info('-----train----')
                # logger.info('epoch %d, step %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,step,training_loss/evaluation_step,recall,precise,span_f1))
                logger.info('epoch %d, global_step %d, loss %.4f'% (epoch,global_step,training_loss/evaluation_steps))
                training_loss=0.0
                #evaluate trainset
                logger.info('-----evaluate train set : ----')
                model.eval()
                recall,precise,span_f1 = evaluate(model,evaluate_data=train_dataloader,span_acc=span_acc)
                model.train()
                logger.info('epoch %d, global_step %d, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,global_step,recall,precise,span_f1))

                #evaluate testset
                logger.info('-----evaluate test set : ----')
                model.eval()
                recall,precise,span_f1 = evaluate(model,evaluate_data=test_dataloader,span_acc=span_acc)
                model.train()
                logger.info('epoch %d, global_step %d, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,global_step,recall,precise,span_f1))
                vis.line([span_f1],[global_step],win=f'f1',
                        opts=dict(title=f'F1', xlabel='step', ylabel='f1', markers=True, markersize=10), update='append')
                vis.line([recall],[global_step],win=f'recall',
                        opts=dict(title=f'Recall', xlabel='step', ylabel='recall', markers=True, markersize=10), update='append')
                vis.line([precise],[global_step],win=f'precise',
                        opts=dict(title=f'precise', xlabel='step', ylabel='precise', markers=True, markersize=10), update='append')
                if best < span_f1:
                    best=span_f1
                    torch.save(model.state_dict(), f=args.ckpt)
                    logger.info('-----save the best model----')  

                model.zero_grad()
                model.train()


def evaluate(model,evaluate_data,span_acc):
    if model.training:
        model.eval()

    count=0
    span_f1=0
    recall=0
    precise=0

    with torch.no_grad():
        for item in evaluate_data:
            count+=1
            input_ids, attention_mask, token_type_ids = item["input_ids"], item["attention_mask"], item["token_type_ids"]
            span_label,span_mask = item['span_label'],item["span_mask"]

            ner_span_logits = model( 
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                ) 

            tmp_recall,tmp_precise,tmp_span_f1=span_acc(logits=ner_span_logits,labels=span_label.to(device),span_mask=span_mask.to(device))
            
            span_f1+=tmp_span_f1
            recall+=tmp_recall
            precise+=tmp_precise

    span_f1 = span_f1/count
    recall=recall/count
    precise=precise/count
    
    return recall,precise,span_f1

def predict(model,test_dataloader,tag_encoder,device):
    logger.info("Evaluating the model...")
    if model.training:
        model.eval()
    
    predictions=[]
    for batch in test_dataloader:
        batch=batch_to_device(inputs=batch,device=device)
        input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        with torch.no_grad():
            outputs=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)#(batch_size,seq_length,num_classes)

        for i in range(outputs.shape[0]):
            indices = torch.argmax(outputs[i],dim=1)#(seq_length,)
            preds = tag_encoder.inverse_transform(indices.cpu().numpy())#(seq_length,)
            preds = [prediction for prediction, offset in zip(preds.tolist(), batch.get('offsets')[i]) if offset]#offsets = [1] + offsets + [1]
            preds = preds[1:-1]
            #print(sum(attention_mask[i])-2,len(preds),preds)
            predictions.append(preds)
    
    return predictions


def main():
    parser = argparse.ArgumentParser()
    # input and output parameters
    parser.add_argument('--model_name_or_path', default = '/home/xhsun/NLP/huggingfaceModels/English/bert-base-uncased', help='path to the BERT')
    parser.add_argument('--file_path', default='/home/xhsun/Desktop/NER_Parsing/pcode/data/conll03', help='path to the ner data')
    parser.add_argument("--ner_train_path", type=str, default="/home/xhsun/Desktop/NER_Parsing/pcode/data/conll03/train.txt",help="train file")
    parser.add_argument("--ner_test_path", type=str, default="/home/xhsun/Desktop/NER_Parsing/pcode/data/conll03/test.txt",help="test file")
    parser.add_argument('--save_dir', default=save_dir, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = '/home/xhsun/Desktop/NER_Parsing/train_models/biaffine/pytorch_model.bin',help='Fine tuned model')
    parser.add_argument("--max_length", type=int, default=128,help="max_length")
    # training parameters
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--lstm_layer', default=1, type=int)
    
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--project_dropout', default=0.2, type = int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--lstm_hidden_size', default=150, type=int)
    parser.add_argument('--to_biaffine_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--max_len', default=196, type = int)
    parser.add_argument('--patience', default=10, type = int)

    #Other parameters
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--num_workers', default=1, type = int)
    args = parser.parse_args()

    ner_label2id=generate_label2id(file_path=args.ner_train_path)
    ner_label2id=process_nerlabel(label2id=ner_label2id)
    tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataloader=yield_data(args=args,file_path=args.ner_train_path,tokenizer=tokenizer,label2id=ner_label2id,is_training=True)
    test_dataloader=yield_data(args=args,file_path=args.ner_test_path,tokenizer=tokenizer,label2id=ner_label2id,is_training=False)

    # args display
    args_config={}
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))
        args_config[k]=v

    with open(os.path.join(save_dir,'args_config.dict'),'w') as f:
        json.dump(args_config,f,ensure_ascii=False)

    train(args,
        ner_label2id,
        train_dataloader,
        test_dataloader,
        load_finetune=False)

if __name__=="__main__":
    main()
