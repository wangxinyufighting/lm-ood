import ast
import umap
import umap.plot
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig

from utils.dataset_utils import DatasetUtil
from utils.startup import logger, exp_configs
from utils.model_utils import load_model, load_tokenizer
from model import RobertaForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification, get_sent_embeddings
# from src.utils.dataset_utils import DatasetUtil
# from src.utils.startup import logger, exp_configs
# from src.utils.model_utils import load_model, load_tokenizer
# from src.model import RobertaForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification, get_sent_embeddings


task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20newsgroups': 20,
    'news-category-modified': 17,
    'clinc150': 150,
}


def save_final_layer_embeddings(processed_dataset, data_util, model, is_id, batch_size=256):
    """

    :param processed_dataset:
    :param data_util:
    :param is_id:
    :param model:
    :return:
    """

    labels = processed_dataset['label']
    dataset_name = data_util.dataset_name

    if exp_configs.debug_mode:
        batch_size = 1

    dataloader = data_util.get_dataloader(processed_dataset, batch_size=batch_size)
    logger.info(f'{len(dataloader)} batches ({len(processed_dataset)} points with batch size {batch_size})')

    del processed_dataset

    final_hidden_state = []
    for i, batch in enumerate(dataloader):
        model.zero_grad()
        if exp_configs.debug_mode and i > 5:
            break

        if i % 50 == 0:
            logger.info(f'Batch {i}')
        with torch.no_grad():
            batch = {k: v.to(exp_configs.device) for k, v in batch.items()}
            _, pooled = get_sent_embeddings(model, batch['input_ids'], batch['attention_mask'])
            final_hidden_state.append(pooled.data.cpu().numpy())
        torch.cuda.empty_cache()

    final_hidden_state = np.concatenate(final_hidden_state)

    N, D = final_hidden_state.shape

    if exp_configs.debug_mode:
        labels = labels[:N]

    if is_id:
        np.savez_compressed(f'{exp_configs.temp_output_dir}/final_hidden_state_{dataset_name}_id.npz', final_hidden_state)
        np.savez_compressed(f'{exp_configs.temp_output_dir}/labels_{dataset_name}_id.npz', labels)
    else:
        np.savez_compressed(f'{exp_configs.temp_output_dir}/final_hidden_state_{dataset_name}_ood.npz', final_hidden_state)
        np.savez_compressed(f'{exp_configs.temp_output_dir}/labels_{dataset_name}_ood.npz', labels)
    logger.info('Saved embeddings.')


def umap_visualization(x_dict, filename, metric='cosine', min_dist=0.1, n_neighbors=10, densmap=True):
    '''
    Applies a UMap Visualization on the tensors of the given dataset. Similar to t-SNE, but better and faster.
    Ref: https://github.com/lmcinnes/umap
    :param x: Typically dataset['input_ids']/ layer embeddings (N, L, D). Must be numpy array.
    :param y: Typically dataset['label']. Must be numpy array.
    :param filename: Filename to save plots with.
    :param metric: 'correlation' or 'cosine' work better. Default is 'euclidean'
    :param min_dist:
    :param n_neighbors:
    :param densmap:
    :return:
    '''

    logger.info('Starting UMap visualization...')
    id_dataset_name = list(x_dict.keys())[0]

    x_test, y_test = [], []
    for k, v in x_dict.items():
        if k != id_dataset_name:
            x_test.append(v)
            y_test.extend([k] * len(v))

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.array(y_test)
    logger.info(f'x: {x_test.shape} y: {y_test.shape}')

    embedding = umap.UMAP(densmap=densmap, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x_dict[id_dataset_name])
    umap.plot.points(embedding, labels=np.array([id_dataset_name] * len(x_dict[id_dataset_name])))
    savename = f'{exp_configs.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}_train-only.png'
    plt.title('Training Data Only')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    logger.info(f'Saved plot to {savename}')

    labels = list(set(y_test))
    colours = ['blue', 'green', 'yellow', 'red']
    colour_key = {labels[i]:colours[i] for i in range(len(labels))}

    embedding = umap.UMAP(densmap=densmap, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x_test)
    umap.plot.points(embedding, labels=y_test, color_key=colour_key)
    savename = f'{exp_configs.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}_test-only.png'
    plt.title('ID and OOD Test Data Only')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    logger.info(f'Saved plot to {savename}')


def get_umap_plots_from_embeddings(dataset_names, filename=None):
    final_hidden_state = {}
    id_dataset_name = dataset_names[0]
    if filename is None:
        filename = id_dataset_name

    final_hidden_state[f'train_{id_dataset_name}'] = np.load(f'{exp_configs.temp_output_dir}/final_hidden_state_{id_dataset_name}_id.npz')['arr_0']

    for ood_dataset_name in dataset_names[1:]:
        key = f'id_{ood_dataset_name}' if ood_dataset_name == id_dataset_name else f'ood_{ood_dataset_name}'
        final_hidden_state[key] = np.load(f'{exp_configs.temp_output_dir}/final_hidden_state_{ood_dataset_name}_ood.npz')['arr_0']

    umap_visualization(x_dict=final_hidden_state, filename=filename)
    logger.info('Done.')


def process_dataset(dataset_name, is_id, tokenizer, split_idx=None):

    data_util = DatasetUtil(dataset_name=dataset_name, max_length=exp_configs.max_seq_length, tokenizer=tokenizer)
    dataset = data_util.get_dataset(dataset_name, split=None)

    if split_idx is None:
        split_idx = 0 if is_id else 1

    logger.info(f'******************** dataset_name:{dataset_name}')
    logger.info(f'******************** dataset_name:{split_idx}')

    dataset_split = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][split_idx]])
    return dataset_split, data_util


if __name__ == "__main__":
    id_dataset = exp_configs.task_name
    ood_datasets = exp_configs.ood_datasets
    model_class = exp_configs.model_class
    model_path = exp_configs.model_name_or_path

    # tokenizer = load_tokenizer('gpt2')
    tokenizer = load_tokenizer(model_str=model_class)
    num_labels = task_to_labels[id_dataset]
    config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
    config.layer_representation_for_ood = 'classifier_input'

    if model_class == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    elif model_class == 'gpt2':
        config.pad_token_id = config.eos_token_id
        config.use_cache = False
        config.sentence_embedding = exp_configs.sentence_embedding
        config.hidden_size = config.n_embd
        config.hidden_dropout_prob = 0.1
        model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
    elif model_class == 't5':
        config.sentence_embedding = exp_configs.sentence_embedding
        config.hidden_size = config.d_model
        config.hidden_dropout_prob = config.dropout_rate
        model = T5ForSequenceClassification.from_pretrained(model_path, config=config)
    model = model.to(exp_configs.device)

    sentence = 'hide new secretions from the parental units'
    sentence_tokenize = tokenizer(sentence)
    input_ids = torch.tensor(sentence_tokenize['input_ids'])
    attention_mask = torch.tensor(sentence_tokenize['attention_mask'])
    input_ids_gpu = input_ids.to(exp_configs.device)
    attention_mask_gpu = attention_mask.to(exp_configs.device)

    output = model.transformer(input_ids_gpu, attention_mask_gpu)
    print(output)


    # id_dataset_tensors, id_data_util = process_dataset(id_dataset, is_id=True, tokenizer=tokenizer)
    # save_final_layer_embeddings(processed_dataset=id_dataset_tensors, data_util=id_data_util, is_id=True, model=model)

    # logger.info(f'******************** ood_datasets:{ood_datasets}')

    # ood_datasets = ast.literal_eval(ood_datasets)
    # for ood_dataset in ood_datasets:
    #     logger.info(f'******************** ood_datasets:{ood_dataset}')
    #     ood_dataset_tensors, ood_data_util = process_dataset(ood_dataset, is_id=False, tokenizer=tokenizer)
    #     save_final_layer_embeddings(processed_dataset=ood_dataset_tensors, data_util=ood_data_util, is_id=False, model=model)

    # get_umap_plots_from_embeddings([id_dataset] + ood_datasets, filename=f'{model_class}')
