import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import ssl
from typing import Callable
import torch
from seqeval.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score
import logging
logger=logging.getLogger('main.utils')

def download_unzip(url_zip: str,
                   dir_extract: str) -> str:
    """Download and unzip a ZIP archive to folder.
    Loads a ZIP file from URL and extracts all of the files to a 
    given folder. Does not save the ZIP file itself.
    Args:
        url_zip (str): URL to ZIP file.
        dir_extract (str): Directory where files are extracted.
    Returns:
        str: a message telling, if the archive was succesfully
        extracted. Obviously the files in the ZIP archive are
        extracted to the desired directory as a side-effect.
    """
    
    # suppress ssl certification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f'Reading {url_zip}')
    with urlopen(url_zip, context=ctx) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'


def download_conll_data(dir: str = None) -> str:
    """Download CoNLL-2003 English data set.
    Downloads the [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) 
    English data set annotated for Named Entity Recognition.
    Args:
        dir (str, optional): Directory where CoNLL-2003 datasets will be saved. If no directory is provided, data will be saved to a hidden folder '.dane' in your home directory.  
                           
    Returns:
        str: a message telling, if the archive was in fact 
        succesfully extracted. Obviously the CoNLL datasets are
        extracted to the desired directory as a side-effect.
    
    Examples:
        >>> download_conll_data()
        >>> download_conll_data(dir = 'conll')
        
    """
    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.conll')

    return download_unzip(url_zip = 'https://data.deepai.org/conll2003.zip',
                          dir_extract = dir)

def match_kwargs(function: Callable, **kwargs) -> dict:
    """Matches Arguments with Function
    Match keywords arguments with the arguments of a function.
    Args:
        function (function): Function to match arguments for.
        kwargs: keyword arguments to match against.
    Returns:
        dict: dictionary with matching arguments and their
        respective values.
    """
    arg_count = function.__code__.co_argcount#14
    args = function.__code__.co_varnames[:arg_count]#'self', 'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 'head_mask', 'inputs_embeds'

    args_dict = {}
    for k, v in kwargs.items():
        if k in args:
            args_dict[k] = v

    return args_dict

def get_ent_tags(all_tags):
    ent_tags=set()
    for each_tag_sequence in all_tags:
        for each_tag in each_tag_sequence:
            ent_tags.add(each_tag)
    return list(ent_tags)

def batch_to_device(inputs,device):
    for key in inputs.keys():
        if type(inputs[key])==list:
            inputs[key]=torch.LongTensor(inputs[key])
        inputs[key]=inputs[key].to(device)
        
    return inputs

def compute_loss(preds, target_tags, masks, device, n_tags):
    
    # initialize loss function.
    lfn = torch.nn.CrossEntropyLoss()

    # Compute active loss to not compute loss of paddings
    active_loss = masks.view(-1) == 1

    active_logits = preds.view(-1, n_tags)
    active_labels = torch.where(
        active_loss,
        target_tags.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target_tags)
    )

    active_labels = torch.as_tensor(active_labels, device = torch.device(device), dtype = torch.long)
    
    # Only compute loss on actual token predictions
    loss = lfn(active_logits, active_labels)

    return loss

def compute_f1(pred_tags,golden_tags,from_test=False):
    assert len(pred_tags)==len(golden_tags)
    count=0
    for pred,golden in zip(pred_tags,golden_tags):
        try:
            assert len(pred)==len(golden)
        except:
            print(len(pred),len(golden))
            print(count)
            raise Exception('length is not consistent!')
        count+=1
        
    result=classification_report(y_pred=pred_tags, y_true=golden_tags, digits=4)
    f1=f1_score(y_pred=pred_tags, y_true=golden_tags)
    acc=accuracy_score(y_pred=pred_tags, y_true=golden_tags)
    precision=precision_score(y_pred=pred_tags, y_true=golden_tags)
    recall=recall_score(y_pred=pred_tags, y_true=golden_tags)

    if from_test==False:
        logger.info('\n'+result)
        logger.info("F1 : {}, accuracy : {}, precision : {}, recall : {}".format(f1,acc,precision,recall))
        return f1
    else:
        print(result)
        print("F1 : {}, accuracy : {}, precision : {}, recall : {}".format(f1,acc,precision,recall))
        return f1