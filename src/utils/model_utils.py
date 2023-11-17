from utils.startup import exp_configs
# from src.utils.startup import exp_configs
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig


def load_model(model_path, num_labels=None, load_only_body=False):

        # model_path = exp_configs.MODEL_DIR + f'/{model_name}'
        if num_labels is not None:
            model_config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, output_hidden_states=True)
        else:
            model_config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        if load_only_body:
            model = AutoModelForMaskedLM.from_pretrained(model_path, config=model_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)
        return model


def load_tokenizer(model_str, ft_model_path):
    if model_str == 'roberta':
        model_name = 'roberta-base'
    elif model_str == 'gpt2':
        model_name = 'gpt2'  # gpt2-small; renamed to base for convenience
        # model_name = '/pretrained_models/gpt2_base'  # gpt2-small; renamed to base for convenience
    elif model_str == 't5':
        model_name = 't5_base'

    if model_str == 'gpt2_ft':
       tokenizer = AutoTokenizer.from_pretrained(ft_model_path, use_fast=True) 
    else:
        # model_path = exp_configs.MODEL_DIR + f'/{model_name}'
        # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if 'gpt2' in model_str:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    return tokenizer
