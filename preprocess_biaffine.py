import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import logging
logger=logging.getLogger('bilstm-biaffine.preprocess_biaffine')


def load_data(file_path):
    #logger.info("raw tokenizer len: {}".format(len(tokenizer)))    
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    sentences = []
    relations = []
    text = []
    relation = []
    idx=1
    for line in lines:
        if line == '\n' or line=='':
            #print(len(text))
            #sentence = (' ').join(text)
            if relation!=[]:
                sentences.append(text)
                relations.append(relation)
            #print(sentence)
            #print(relation)
            text = []
            relation = []
            idx=1
            continue
        line = line.strip().split('\t')
        assert len(line)==2
        word, rel = line[0], line[-1].strip()
        #text.append(tokenizer.tokenize(word)[0])
        text.append(word)
        relation.append((int(idx),int(idx),rel))
        idx+=1

    #logger.info("new tokenizer len: {}".format(len(tokenizer)))  
    logger.info("File {} has {} sentences!".format(file_path,len(sentences)))
    all_sen_lengths=[len(sen_list) for sen_list in sentences]
    logger.info("Average sentence length : {}".format(sum(all_sen_lengths)/len(all_sen_lengths)))
    return sentences, relations

#encode_sent is input_ids, span_label.shape==(max_length,max_length), 只有在两个token之间的span是一个实体类型的case下，对应位置是1
#span_mask.shape==(max_length,max_length),只有左上半部分是1,pad位置是0

def get_span_mask_label(args,sentence,tokenizer,attention_mask,relation,label2id):
    assert type(sentence)==list
    
    zero = [0 for i in range(args.max_length)]
    span_mask=[ attention_mask for i in range(sum(attention_mask))]
    span_mask.extend([ zero for i in range(sum(attention_mask),args.max_length)])
    #span_mask=np.triu(np.array(span_mask)).tolist()#将下三角全部置0

    span_label = [0 for i in range(args.max_length)]#label2id['O']=0
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    ner_relation = []
    assert len(sentence)==len(relation)

    new_relation=[]
    idx=1
    for i in range(len(sentence)):
        _,_,tag=relation[i]
        wordpiece=tokenizer.tokenize(sentence[i])
        if len(wordpiece)==1:
            new_relation.append((idx,idx,tag))
            idx+=1
        else:
            for j in range(len(wordpiece)):
                cur_tag=tag
                if j>0:
                    cur_tag='I-'+tag.split('-')[1] if tag!='O' else tag
                new_relation.append((idx,idx,cur_tag))
                idx+=1
                if idx==args.max_length-2:
                    break
        if idx==args.max_length-2:
            break

    relation=new_relation

    ner_relation = []
    start_idx = 0
    end_idx = 0
    pre_label = 'O'
    #relabelling
    ent_tag='O'
    relation.append((relation[-1][0]+1,relation[-1][0]+1,'O'))
    for i, (idx,_,cur_label)in enumerate(relation):

        if cur_label[0]=='O':
            if pre_label[0]!='O' and pre_label[0]!='S':
                ner_relation.append((start_idx,idx-1,ent_tag))
                start_idx=idx

        if cur_label[0]=='B':
            if pre_label[0]=='O' or pre_label[0]=='S':
                start_idx=idx
                ent_tag=cur_label[2:]
            if pre_label[0]=='I' or pre_label[0]=='E':
                ner_relation.append((start_idx,idx-1,ent_tag))
                start_idx=idx
                ent_tag=cur_label[2:]

        pre_label=cur_label

        
    for start_idx, end_idx, rel in ner_relation:
        #输入是wordpiece形式，但是输入的标签不能是wordpiece形式的
        span_label[start_idx, end_idx] = label2id[rel+'_whole']#label2id形如：{‘0’：0,‘ORG_whole’:1,'...}

    return span_mask,span_label,ner_relation

def convert_example_to_feature(args,tokenizer,sentence,labels,label2id):
    '''
    sentence is a list, 每一个元素是一个word
    labels is a list, 每一个元素是一个tuple，包含(id,id,tag)，id指的是对应word的实体tag
    '''
    assert len(sentence)==len(labels)
    tokens = []
    offsets = []

    for i, word in enumerate(sentence):
        # bert tokenization
        wordpieces = tokenizer.tokenize(word)
        tokens.extend(wordpieces)
        # make room for CLS if there is an identified word piece
        if len(wordpieces)>0:
            offsets.extend([1]+[0]*(len(wordpieces)-1))

    tokens=tokens[:args.max_length-2]
    offsets=offsets[:args.max_length-2]

    input_ids=tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids) 
    offsets = [1] + offsets + [1]#1代表这个位置是实际的词，0代表这个位置的词与前面的词是一个单词的

    padding_len = args.max_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_len)
    attention_mask = attention_mask + ([0] * padding_len)
    offsets = offsets + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)
    span_mask,span_label,ner_relation=get_span_mask_label(args=args,
                                                        sentence=sentence,
                                                        tokenizer=tokenizer,
                                                        attention_mask=attention_mask,
                                                        relation=labels,
                                                        label2id=label2id)
    return input_ids,attention_mask,offsets,token_type_ids,span_mask,span_label,ner_relation,tokens


def data_pre(args,file_path, tokenizer, label2id):
    sentences, relations = load_data(file_path=file_path)
    data = []

    for i,sentence in tqdm(enumerate(sentences)):
        label=relations[i]
        input_ids,attention_mask,offsets,token_type_ids,span_mask,span_label,ner_relation, wordpiece_tokens=convert_example_to_feature(args,tokenizer=tokenizer,sentence=sentence,labels=label,label2id=label2id)

        tmp = {}
        tmp['input_ids'] = input_ids
        tmp['token_type_ids'] = token_type_ids
        tmp['attention_mask'] = attention_mask
        tmp['span_label'] = span_label
        tmp['span_mask'] = span_mask
        tmp['input_tokens'] = sentence#list
        tmp['span_tokens'] = label#list
        tmp['converted_span']=ner_relation
        tmp['offsets']=offsets
        tmp['wordpiece_tokens']=wordpiece_tokens
        # wordpiece=[]
        # for start_id,end_id,tag in ner_relation:
        #     wordpiece.append(' '.join(tokenizer.convert_ids_to_tokens(batch_input_ids[k][start_id:end_id+1])))
        # tmp['wordpiece']=wordpiece
        data.append(tmp)

    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "token_type_ids": torch.tensor(item['token_type_ids']).long(),
            "attention_mask": torch.tensor(item['attention_mask']).float(),
            "span_label": torch.tensor(item['span_label']).long(),
            "span_mask": torch.tensor(item['span_mask']).long(),
            "offsets": torch.tensor(item['offsets']).long()
        }
        return one_data

def yield_data(args,file_path, tokenizer, label2id, is_training=True):
    data = data_pre(args, file_path, tokenizer, label2id=label2id)
    logger.info("number of examples : {}".format(len(data)))
    logger.info("Printing some exampls...")
    for _ in range(3):
        idx=random.randint(a=0,b=len(data)-1)
        example=data[idx]
        input_ids=example['input_ids']
        token_type_ids=example['token_type_ids']
        attention_mask=example['attention_mask']
        span_label=example['span_label']
        span_mask=example['span_mask']
        input_tokens=example['input_tokens']
        span_tokens=example['span_tokens']
        offsets=example['offsets']
        wordpiece_tokens=example['wordpiece_tokens']
        logger.info("input tokens(sentence) : {}".format(input_tokens))
        logger.info("wordpiece tokens(wordpiece) : {}".format(wordpiece_tokens))
        logger.info("span tokens(label) : {}".format(span_tokens))
        logger.info("input_ids : {}".format(' '.join([str(i) for i in input_ids])))
        logger.info("attention_mask : {}".format(' '.join([str(i) for i in attention_mask])))

        length=sum(attention_mask)
        print_span_mask=np.array(span_mask)[:length+2,:length+2]
        print_span_label=np.array(span_label)[:length+2,:length+2]
        # logger.info("Span mask : ")
        # for row in print_span_mask:
        #     logger.info(" ".join([str(i) for i in row]))
        # logger.info("Span label(wordpiece) : ")
        # for row in print_span_label:
        #     logger.info(" ".join([str(i) for i in row])) 

        logger.info("converted_span (wordpiece): {}".format(example['converted_span']))
        # logger.info("span_label : {}".format(' '.join([str(i) for i in span_label])))
        # logger.info("span_mask : {}".format(' '.join([str(i) for i in span_mask])))
        logger.info("input length: {}".format(len(input_ids)))
        logger.info('-'*100)

    logger.info("label2id : {}".format(label2id))
    dataset=MyDataset(data=data)
    if is_training:
        return DataLoader(dataset, batch_size=args.train_batch_size, shuffle=is_training, num_workers=args.num_workers)
    else:
        return DataLoader(dataset, batch_size=args.test_batch_size, shuffle=is_training, num_workers=args.num_workers)

def generate_label2id(file_path):
    with open(file_path) as f:
        lines=f.readlines()
    label2id={}
    for line in lines:
        line_split=line.strip().split()
        if len(line_split)>1:
            label2id[line_split[-1]]=len(label2id)

    return label2id

def process_nerlabel(label2id):
    #label2id,id2label,num_labels = tools.load_schema_ner()
    #Since different ner dataset has different entity categories, it is inappropriate to pre-assign entity labels
    new_={}
    new_={'O':0}
    for label in label2id:
        if label!='O':
            label=label.split('-')[1]
            if label+'_whole' not in new_:
                new_[label+'_whole']=len(new_)
    return new_