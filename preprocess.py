from sklearn.utils import shuffle
import torch
import warnings
from itertools import compress
import os
import csv
import transformers
import sklearn.preprocessing

def get_span_mask_label(args,sentence,tokenizer,attention_mask,relation,label2id,mode):
    zero = [0 for i in range(args.max_length)]
    span_mask=[ attention_mask for i in range(sum(attention_mask))]
    span_mask.extend([ zero for i in range(sum(attention_mask),args.max_length)])
    #span_mask=np.triu(np.array(span_mask)).tolist()#将下三角全部置0

    span_label = [0 for i in range(args.max_length)]#label2id['O']=0
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    ner_relation = []
    sentence=sentence.split(' ')
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

    if mode == 'dp_train' or mode == 'dp_test':            
        for start_idx, end_idx, rel in relation:
            #print(start_idx, end_idx, rel)
            span_label[start_idx, end_idx] = label2id[rel]
    elif mode == 'ner_train' or mode == 'ner_test':
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
            if mode == 'ner_train':
                for i in range(start_idx, end_idx+1):
                    for j in range(start_idx, end_idx+1):
                        span_label[i, j] = label2id[rel+'_sub']
            #输入是wordpiece形式，但是输入的标签不能是wordpiece形式的
            span_label[start_idx, end_idx] = label2id[rel+'_whole']
    return span_mask,span_label,ner_relation

class DataSet():
    def __init__(self, 
                sentences: list, 
                tags: list, 
                transformer_tokenizer: transformers.PreTrainedTokenizer, 
                transformer_config: transformers.PretrainedConfig, 
                max_len: int, 
                tag_encoder: sklearn.preprocessing.LabelEncoder, 
                tag_outside: str,
                take_longest_token: bool = True,
                pad_sequences : bool = True) -> None:
        """Initialize DataSetReader
        Initializes DataSetReader that prepares and preprocesses 
        DataSet for Named-Entity Recognition Task and training.
        Args:
            sentences (list): Sentences.
            tags (list): Named-Entity tags.
            transformer_tokenizer (transformers.PreTrainedTokenizer): 
                tokenizer for transformer.
            transformer_config (transformers.PretrainedConfig): Config
                for transformer model.
            max_len (int): Maximum length of sentences after applying
                transformer tokenizer.
            tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
                for Named-Entity tags.
            tag_outside (str): Special Outside tag. like 'O'
            pad_sequences (bool): Pad sequences to max_len. Defaults
                to True.
        """
        self.sentences = sentences
        self.tags = tags
        self.transformer_tokenizer = transformer_tokenizer
        self.max_len = max_len
        self.tag_encoder = tag_encoder
        self.pad_token_id = transformer_config.pad_token_id
        self.tag_outside_transformed = tag_encoder.transform([tag_outside])[0]
        self.take_longest_token = take_longest_token
        self.pad_sequences = pad_sequences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        tags = self.tags[item]
        # encode tags
        tags = self.tag_encoder.transform(tags)
        
        # check inputs for consistancy
        assert len(sentence) == len(tags)

        input_ids = []
        target_tags = []
        tokens = []
        offsets = []
        
        # for debugging purposes
        # print(item)
        for i, word in enumerate(sentence):
            # bert tokenization
            wordpieces = self.transformer_tokenizer.tokenize(word)
            if self.take_longest_token:
                piece_token_lengths=[len(token) for token in wordpieces]
                word=wordpieces[piece_token_lengths.index(max(piece_token_lengths))]
                wordpieces=[word]#仅仅取最长的token

            tokens.extend(wordpieces)
            # make room for CLS if there is an identified word piece
            if len(wordpieces)>0:
                offsets.extend([1]+[0]*(len(wordpieces)-1))
            # Extends the ner_tag if the word has been split by the wordpiece tokenizer
            target_tags.extend([tags[i]] * len(wordpieces)) 
        
        if self.take_longest_token:
            assert len(tokens)==len(sentence)==len(target_tags)
        # Make room for adding special tokens (one for both 'CLS' and 'SEP' special tokens)
        # max_len includes _all_ tokens.
        if len(tokens) > self.max_len-2:
            msg = f'Sentence #{item} length {len(tokens)} exceeds max_len {self.max_len} and has been truncated'
            warnings.warn(msg)
        tokens = tokens[:self.max_len-2] 
        target_tags = target_tags[:self.max_len-2]
        offsets = offsets[:self.max_len-2]

        # encode tokens for BERT
        # TO DO: prettify this.
        input_ids = self.transformer_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.transformer_tokenizer.cls_token_id] + input_ids + [self.transformer_tokenizer.sep_token_id]
        
        # fill out other inputs for model.    
        target_tags = [self.tag_outside_transformed] + target_tags + [self.tag_outside_transformed] 
        attention_mask = [1] * len(input_ids)
        # set to 0, because we are not doing NSP or QA type task (across multiple sentences)
        # token_type_ids distinguishes sentences.
        token_type_ids = [0] * len(input_ids) 
        offsets = [1] + offsets + [1]

        # Padding to max length 
        # compute padding length
        if self.pad_sequences:
            padding_len = self.max_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token_id] * padding_len)
            attention_mask = attention_mask + ([0] * padding_len)  
            offsets = offsets + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_tags = target_tags + ([self.tag_outside_transformed] * padding_len)  
    
        return {'input_ids' : torch.tensor(input_ids, dtype = torch.long),
                'attention_mask' : torch.tensor(attention_mask, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target_tags' : torch.tensor(target_tags, dtype = torch.long),
                'offsets': torch.tensor(offsets, dtype = torch.long)} 
      
def create_dataloader(sentences, 
                      tags, 
                      transformer_tokenizer, 
                      transformer_config, 
                      max_len,  
                      tag_encoder, 
                      tag_outside,
                      batch_size = 1,
                      num_workers = 1,
                      take_longest_token=True,
                      pad_sequences = True,
                      is_training = True):

    if not pad_sequences and batch_size > 1:
        print("setting pad_sequences to True, because batch_size is more than one.")
        pad_sequences = True

    data_reader = DataSet(
        sentences = sentences, 
        tags = tags,
        transformer_tokenizer = transformer_tokenizer, 
        transformer_config = transformer_config,
        max_len = max_len,
        tag_encoder = tag_encoder,
        tag_outside = tag_outside,
        take_longest_token = take_longest_token,
        pad_sequences = pad_sequences)
        # Don't pad sequences if batch size == 1. This improves performance.

    data_loader = torch.utils.data.DataLoader(
        data_reader, batch_size = batch_size, num_workers = num_workers, shuffle=is_training
    )

    return data_loader

def get_conll_data(split: str = 'train', 
                   limit_length: int = 196, 
                   dir: str = None) -> dict:
    assert isinstance(split, str)
    splits = ['train', 'dev', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.

    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'
    
    file_path = os.path.join(dir, f'{split}.txt')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'

    # read data from file.
    with open(file_path, 'r') as f:
        lines=f.readlines()

    sentences = []
    sentence = []
    entities = []
    entity = []

    for line in lines:

        if line == '\n' or line=='':
            #print(len(text))
            if entity!=[]:
                sentences.append(sentence)
                entities.append(entity)
            entity=[]
            sentence=[]
            continue
        line = line.strip().split()
        assert len(line)==2
        sentence.append(line[0])
        entity.append(line[1])
    
    return {'sentences': sentences, 'tags': entities}

def get_semeval_data(split: str = 'train', 
                   limit_length: int = 196, 
                   dir: str = None,
                   word_idx=1,
                   entity_idx=4) -> dict:
    assert isinstance(split, str)
    splits = ['train', 'dev', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.

    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'
    
    file_path = os.path.join(dir, f'{split}.txt')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'

    # read data from file.
    with open(file_path, 'r') as f:
        lines=f.readlines()

    sentences = []
    sentence = []
    entities = []
    entity = []

    for line in lines:

        if line == '\n' or line=='':
            #print(len(text))
            if entity!=[]:
                sentences.append(sentence)
                entities.append(entity)
            entity=[]
            sentence=[]
            continue
        line = line.strip().split()
        sentence.append(line[word_idx])
        entity.append(line[entity_idx])
    
    return {'sentences': sentences, 'tags': entities}