import numpy as np
import torch
import argparse
import os,json
import sys
from tqdm import tqdm
import sklearn.preprocessing
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import transformers
import random
from preprocess import create_dataloader, get_semeval_data
from utils import compute_loss, get_ent_tags, batch_to_device, compute_f1
from models.bert_lstm_crf import NERNetwork
import logging

logger=logging.getLogger('main')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log/log.txt',mode='w')
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

device='cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device {}".format(device))

def train(args,
        train_dataloader,
        tag_encoder,
        train_conll_tags,
        test_conll_tags,
        test_dataloader):
    
    n_tags=tag_encoder.classes_.shape[0]
    logger.info("n_tags : {}".format(n_tags))

    print_loss_step=len(train_dataloader)//5
    evaluation_steps=len(train_dataloader)//2
    logger.info("Under an epoch, loss will be output every {} step, and the model will be evaluated every {} step".format(print_loss_step,evaluation_steps))


    model=NERNetwork(args,n_tags=n_tags)
    if args.ckpt is not None:
        load_result=model.load_state_dict(torch.load(args.ckpt,map_location='cpu'),strict=False)
        logger.info("Load ckpt to continue training !")
        logger.info("missing and unexcepted key : {}".format(str(load_result)))

    model.to(device=device)
    logger.info("Using device : {}".format(device))
    optimizer_parameters=model.parameters()
    optimizer = AdamW(optimizer_parameters, lr = args.learning_rate)
    num_train_steps=int(len(train_conll_tags)//args.train_batch_size//args.gradient_accumulation_steps)*args.epochs
    warmup_steps=int(num_train_steps*args.warmup_proportion)
    logger.info("num_train_steps : {}, warmup_proportion : {}, warmup_steps : {}".format(num_train_steps,args.warmup_proportion,warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )

    global_step=0
    
    previous_f1=-1
    predictions=predict(model=model,test_dataloader=test_dataloader,tag_encoder=tag_encoder,device=device)
    f1=compute_f1(pred_tags=predictions,golden_tags=test_conll_tags)
    if f1>previous_f1:
        logger.info("Previous f1 score is {} and current f1 score is {}".format(previous_f1,f1))
        previous_f1=f1

    for epoch in range(args.epochs):
        model.train()
        model.zero_grad()
        training_loss=0.0
        for iteration, batch in tqdm(enumerate(train_dataloader)):
            batch=batch_to_device(inputs=batch,device=device)
            input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
            loss=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,target_tags=batch['target_tags'])#(batch_size,seq_length,num_classes)
            #target_tags将CLS和SEP赋予标签O
            training_loss+=loss.item()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (iteration+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step+=1

            if (iteration+1) % print_loss_step == 0:
                training_loss/=print_loss_step
                logger.info("Epoch : {}, global_step : {}/{}, loss_value : {} ".format(epoch,global_step,num_train_steps,training_loss))
                training_loss=0.0

            if (iteration+1) % evaluation_steps == 0 :
                predictions=predict(model=model,test_dataloader=test_dataloader,tag_encoder=tag_encoder,device=device)
                f1=compute_f1(pred_tags=predictions,golden_tags=test_conll_tags)
                if f1>previous_f1:
                    torch.save(model.state_dict(),f=os.path.join(args.save_dir,'pytorch_model.bin'))
                    logger.info("Previous f1 score is {} and current f1 score is {}, best model has been saved in {}".format(previous_f1,f1,os.path.join(args.save_dir,'pytorch_model.bin')))
                    previous_f1=f1                    
                    
                else:
                    args.patience-=1
                    logger.info("Left patience is {}".format(args.patience))
                    if args.patience==0:
                        logger.info("Total patience is {}, run our of patience, early stop!".format(args.patience))
                        return

                model.zero_grad()
                model.train()

    # predictions=predict(model=model,test_dataloader=train_dataloader,tag_encoder=tag_encoder,device=device)
    # f1=compute_f1(pred_tags=predictions,golden_tags=train_conll_tags)
    # logger.info("F1 on train set : {}".format(f1))
    # logger.info("Best f1 on test set : {}".format(previous_f1))



def predict(model,test_dataloader,tag_encoder,device):
    logger.info("Evaluating the model...")
    if model.training:
        model.eval()
    
    predictions=[]
    for batch in test_dataloader:
        batch=batch_to_device(inputs=batch,device=device)
        input_ids,attention_mask,token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        with torch.no_grad():
            outputs=model.predict(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)#(batch_size,seq_length,num_classes)

        for i,predict_tag_seq in enumerate(outputs):
            preds = tag_encoder.inverse_transform(predict_tag_seq)#(with wordpiece)
            preds = [prediction for prediction, offset in zip(preds.tolist(), batch.get('offsets')[i]) if offset]#offsets = [1] + offsets + [1]
            preds = preds[1:-1]
            #print(sum(attention_mask[i])-2,len(preds),preds)
            predictions.append(preds)
    
    return predictions


def main():
    parser = argparse.ArgumentParser()
    # input and output parameters
    parser.add_argument('--model_name_or_path', default = 'bert-base-uncased', help='path to the BERT')
    parser.add_argument('--file_path', default='data/conll03', help='path to the ner data')
    parser.add_argument('--save_dir', default='saved_models/', help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None,help='Fine tuned model')
    # training parameters
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--lstm_hidden_size', default=150, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--max_len', default=196, type = int)
    parser.add_argument('--patience', default=100, type = int)

    #Other parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', default=1, type = int)
    parser.add_argument('--take_longest_token', default=False, type = bool)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        logger.info("save_dir not exists, created!")
        os.makedirs(args.save_dir,exist_ok=True)
        
    train_conll_data=get_semeval_data(split='train',dir=args.file_path,word_idx=1,entity_idx=3)
    #test_conll_data=get_conll_data(split='valid',dir=args.file_path)
    test_conll_data=get_semeval_data(split='dev',dir=args.file_path,word_idx=1,entity_idx=3)
    logger.info("train sentences num : {}".format(len(train_conll_data['sentences'])))
    logger.info("test sentences num : {}".format(len(test_conll_data['sentences'])))
    logger.info("Logging some examples...")
    for _ in range(5):
        i=random.randint(0,len(test_conll_data['tags'])-1)
        sen=test_conll_data['sentences'][i]
        ent=test_conll_data['tags'][i]
        for k in range(len(sen)):
            logger.info("{}  {}".format(sen[k],ent[k]))
        logger.info('-'*50)

    tag_scheme=get_ent_tags(all_tags=train_conll_data.get('tags'))
    tag_outside='O'
    if tag_outside in tag_scheme:
        del tag_scheme[tag_scheme.index(tag_outside)]
    tag_complete=[tag_outside]+tag_scheme
    print(tag_complete,len(tag_complete))
    with open(os.path.join(args.save_dir,'label.json'),'w') as f:
        json.dump(obj=' '.join(tag_complete),fp=f)
    logger.info("Tag scheme : {}".format(' '.join(tag_scheme)))
    logger.info("Tag has been saved in {}".format(os.path.join(args.save_dir,'label.json')))
    tag_encoder=sklearn.preprocessing.LabelEncoder()
    tag_encoder.fit(tag_complete)

    #tokenizer_parameters=json.load(open(os.path.join(args.model_name_or_path,'tokenizer_config.json')))
    transformer_tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
    transformer_config=AutoConfig.from_pretrained(args.model_name_or_path)

    train_dataloader=create_dataloader(sentences=train_conll_data.get('sentences'),
                                      tags=train_conll_data.get('tags'),
                                      transformer_tokenizer=transformer_tokenizer,
                                      transformer_config=transformer_config,
                                      max_len=args.max_len,
                                      tag_encoder=tag_encoder,
                                      tag_outside=tag_outside,
                                      batch_size=args.train_batch_size,
                                      num_workers=args.num_workers,
                                      take_longest_token=args.take_longest_token,
                                      is_training=True)
    test_dataloader=create_dataloader(sentences=test_conll_data.get('sentences'),
                                      tags=test_conll_data.get('tags'),
                                      transformer_tokenizer=transformer_tokenizer,
                                      transformer_config=transformer_config,
                                      max_len=args.max_len,
                                      tag_encoder=tag_encoder,
                                      tag_outside=tag_outside,
                                      batch_size=args.test_batch_size,
                                      num_workers=args.num_workers,
                                      take_longest_token=args.take_longest_token,
                                      is_training=False)

    # args display
    args_config={}
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))
        args_config[k]=v

    with open(os.path.join(args.save_dir,'args_config.dict'),'w') as f:
        json.dump(args_config,f,ensure_ascii=False)

    train(args=args,train_dataloader=train_dataloader,
            tag_encoder=tag_encoder,
            train_conll_tags=train_conll_data.get('tags'),
            test_conll_tags=test_conll_data.get('tags'),
            test_dataloader=test_dataloader)


if __name__=="__main__":
    main()
