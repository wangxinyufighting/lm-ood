import os
import ast
import torch
import warnings
import numpy as np
from tqdm import tqdm
from datasets import load_metric, load_dataset
from transformers import RobertaConfig, GPT2Config, T5Config, RobertaModelWithHeads, GPT2LMHeadModel
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model

from evaluation import evaluate_ood
from utils.dataset_utils import DatasetUtil
from utils.model_utils import load_tokenizer
from utils.startup import set_seed, collate_fn, logger, exp_configs
from model import ClassificationHead, RobertaForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification, prepare_ood
# from src.evaluation import evaluate_ood
# from src.utils.dataset_utils import DatasetUtil
# from src.utils.model_utils import load_tokenizer
# from src.utils.startup import set_seed, collate_fn, logger, exp_configs
# from src.model import RobertaForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification, prepare_ood

LABEL_ID = {
                'oos':0
                , "translate":1
                ,"transfer":2
                ,"timer":3
                ,"definition":4
                ,"meaning_of_life":5
                ,"insurance_change":6
                ,"find_phone":7
                ,"travel_alert":8
                ,"pto_request":9
                ,"improve_credit_score":10
                ,"fun_fact":11
                ,"change_language":12
                ,"payday":13
                ,"replacement_card_duration":14
                ,"time":15
                ,"application_status":16
                ,"flight_status":17
                ,"flip_coin":18
                ,"change_user_name":19
                ,"where_are_you_from":20
                ,"shopping_list_update":21
                ,"what_can_i_ask_you":22
                ,"maybe":23
                ,"oil_change_how":24
                ,"restaurant_reservation":25
                ,"balance":26
                ,"confirm_reservation":27
                ,"freeze_account":28
                ,"rollover_401k":29
                ,"who_made_you":30
                ,"distance":31
                ,"user_name":32
                ,"timezone":33
                ,"next_song":34
                ,"transactions":35
                ,"restaurant_suggestion":36
                ,"rewards_balance":37
                ,"pay_bill":38
                ,"spending_history":39
                ,"pto_request_status":40
                ,"credit_score":41
                ,"new_card":42
                ,"lost_luggage":43
                ,"repeat":44
                ,"mpg":45
                ,"oil_change_when":46
                ,"yes":47
                ,"travel_suggestion":48
                ,"insurance":49
                ,"todo_list_update":50
                ,"reminder":51
                ,"change_speed":52
                ,"tire_pressure":53
                ,"no":54
                ,"apr":55
                ,"nutrition_info":56
                ,"calendar":57
                ,"uber":58
                ,"calculator":59
                ,"date":60
                ,"carry_on":61
                ,"pto_used":62
                ,"schedule_maintenance":63
                ,"travel_notification":64
                ,"sync_device":65
                ,"thank_you":66
                ,"roll_dice":67
                ,"food_last":68
                ,"cook_time":69
                ,"reminder_update":70
                ,"report_lost_card":71
                ,"ingredient_substitution":72
                ,"make_call":73
                ,"alarm":74
                ,"todo_list":75
                ,"change_accent":76
                ,"w2":77
                ,"bill_due":78
                ,"calories":79
                ,"damaged_card":80
                ,"restaurant_reviews":81
                ,"routing":82
                ,"do_you_have_pets":83
                ,"schedule_meeting":84
                ,"gas_type":85
                ,"plug_type":86
                ,"tire_change":87
                ,"exchange_rate":88
                ,"next_holiday":89
                ,"change_volume":90
                ,"who_do_you_work_for":91
                ,"credit_limit":92
                ,"how_busy":93
                ,"accept_reservations":94
                ,"order_status":95
                ,"pin_change":96
                ,"goodbye":97
                ,"account_blocked":98
                ,"what_song":99
                ,"international_fees":100
                ,"last_maintenance":101
                ,"meeting_schedule":102
                ,"ingredients_list":103
                ,"report_fraud":104
                ,"measurement_conversion":105
                ,"smart_home":106
                ,"book_hotel":107
                ,"current_location":108
                ,"weather":109
                ,"taxes":110
                ,"min_payment":111
                ,"whisper_mode":112
                ,"cancel":113
                ,"international_visa":114
                ,"vaccines":115
                ,"pto_balance":116
                ,"directions":117
                ,"spelling":118
                ,"greeting":119
                ,"reset_settings":120
                ,"what_is_your_name":121
                ,"direct_deposit":122
                ,"interest_rate":123
                ,"credit_limit_change":124
                ,"what_are_your_hobbies":125
                ,"book_flight":126
                ,"shopping_list":127
                ,"text":128
                ,"bill_balance":129
                ,"share_location":130
                ,"redeem_rewards":131
                ,"play_music":132
                ,"calendar_update":133
                ,"are_you_a_bot":134
                ,"gas":135
                ,"expiration_date":136
                ,"update_playlist":137
                ,"cancel_reservation":138
                ,"tell_joke":139
                ,"change_ai_name":140
                ,"how_old_are_you":141
                ,"car_rental":142
                ,"jump_start":143
                ,"meal_suggestion":144
                ,"recipe":145
                ,"income":146
                ,"order":147
                ,"traffic":148
                ,"order_checks":149
                ,"card_declined":150
            }

warnings.filterwarnings("ignore")

task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20newsgroups': 20,
    'news-category-modified': 17,
    'clinc150': 150,
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20newsgroups': 'mnli',
    'clinc150': 'mnli',
    'news-category-modified': 'mnli'
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):

    def detect_ood():
        print('****** detect_ood start! ')
        prepare_ood(model, dataloader=train_dataloader, is_train=True)
        prepare_ood(model, dataloader=dev_dataloader, is_train=False)
        logger.info(f'On train data, Dispersion={model.dispersion}; Compactness={model.compactness}')

        for tag, ood_features in benchmarks:
            ood_dataloader = torch.utils.data.DataLoader(ood_features, batch_size=args.batch_size, collate_fn=collate_fn)
            results = evaluate_ood(args, model, test_dataloader, ood_dataloader, tag=tag)
            with open('./result_new.txt', 'w') as w:
                for k, v in results.items():
                    logger.info(f'{k}: {v}')
                    w.write(str(k)+':'+str(v)+'\n')

            torch.cuda.empty_cache()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    if not args.epoch_wise_eval:
        detect_ood()

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

def evaluate(args, model, dataloader, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()

        # Adding from my codebase
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
        acc = accuracy_score(labels, preds)
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        result['acc'] = acc

        return result

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["labels"] = None # To ensure loss isn't calculated in model.forward()
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results


def main(args):
    if exp_configs.set_seed:
        set_seed(args)

    num_labels = task_to_labels[args.task_name]
    tokenizer = load_tokenizer(model_str=args.model_class, ft_model_path=args.ft_model_path)

    # Load model config
    if args.model_class == 'roberta':
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    elif 'gpt2' in args.model_class:
        config = GPT2Config.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.pad_token_id = config.eos_token_id
        config.use_cache = False
        config.hidden_size = config.n_embd
        config.hidden_dropout_prob = 0.1
    elif args.model_class == 't5':
        config = T5Config.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.hidden_size = config.d_model
        config.hidden_dropout_prob = config.dropout_rate

    config.layer_representation_for_ood = args.layer_representation_for_ood
    config.gradient_checkpointing = True
    config.contrastive_weight = args.contrastive_weight
    config.cross_entropy_weight = args.cross_entropy_weight
    config.contrastive_loss = args.contrastive_loss
    config.tau = args.tau
    config.report_all_metrics = args.report_all_metrics
    config.sentence_embedding = args.sentence_embedding
    config.add_token_num = args.add_token_num

    # Load model
    if args.model_class == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif 'gpt2' in args.model_class:
        model = GPT2ForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_class == 't5':
        model = T5ForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    # elif 'gpt2_ft' == args.model_class:
    #     model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

    model.to(exp_configs.device)
    logger.info(f'Loaded model. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


    datasets = ast.literal_eval(args.ood_datasets)
    benchmarks = ()

    def process_dataset_split(dataset_split):
        if args.samples_to_keep is not None:
            dataset_split = dataset_split.select(range(args.samples_to_keep))
        dataset_split = data_util.get_tensors_for_finetuning(dataset_split, format='torch')
        dataset_split = dataset_split.rename_column('label', 'labels')
        dataset_split = [x for x in dataset_split]

        processed_dataset_split = []
        for datum in dataset_split:
            if datum['labels'].numpy() == -1:
                datum['labels'] = torch.tensor(0, dtype=torch.int64)  # Following convention of original paper
            processed_dataset_split.append(datum)

        return processed_dataset_split

    for dataset_name in datasets:
        add_prefix_token = args.add_token_num

        def tokenize_function(batch):
            return tokenizer(batch['text'], truncation=True, padding=True, max_length=args.max_seq_length)

        def tokenize_function_new(batch):
            tmp = {k:[' 0'*add_prefix_token+i for i in v] if k=='text' else v for k,v in batch.items()}
            return tokenizer(tmp['text'], truncation=True, padding=True, max_length=args.max_seq_length)

        def label_to_id(batch):
            label = batch["labels"]
            if label == 'oos':
                return {"labels":LABEL_ID[label]}
            else:
                return {"labels":LABEL_ID[label]-1}
        
        def get_dataset_by_split(dataset_raw, split, add_prefix_token):
            dataset_raw_split = dataset_raw[split]
            dataset_raw_split = dataset_raw_split.map(label_to_id) 
            if add_prefix_token > 0:
                dataset_raw_split_tokenized = dataset_raw_split.map(tokenize_function_new, batched=True, batch_size=None)
            else:
                dataset_raw_split_tokenized = dataset_raw_split.map(tokenize_function, batched=True, batch_size=None)

            dataset_raw_split_tokenized = dataset_raw_split_tokenized.remove_columns('text')

            dataset_raw_split_tokenized.set_format("pt")

            return dataset_raw_split_tokenized
        
        if dataset_name == 'clinc150':
            TEST_ID = 'TEST_ID'
            TEST_OOD = 'TEST_OOD'
            
            train_file = '/root/autodl-fs/lm-ood/datasets/clinc150/train.csv'
            test_file = '/root/autodl-fs/lm-ood/datasets/clinc150/test.csv'
            # test_id_file = '/root/autodl-fs/lm-ood/datasets/clinc150/ood_test_id.csv'
            test_ood_file = '/root/autodl-fs/lm-ood/datasets/clinc150/ood_test.csv'
            val_file = '/root/autodl-fs/lm-ood/datasets/clinc150/val.csv'

            # train_file = '/root/autodl-fs/lm-ood/datasets/clinc150/train_small.csv'
            # test_file = '/root/autodl-fs/lm-ood/datasets/clinc150/test_small.csv'
            # # test_id_file = '/root/autodl-fs/lm-ood/datasets/clinc150/ood_test_id.csv'
            # test_ood_file = '/root/autodl-fs/lm-ood/datasets/clinc150/ood_test_small.csv'
            # val_file = '/root/autodl-fs/lm-ood/datasets/clinc150/val_small.csv'

            data_files = {
                    'train': train_file
                    , 'test': test_file
                    , 'val': val_file
                    # , TEST_ID: test_id_file
                    , TEST_OOD: test_ood_file
                }
            dataset_raw = load_dataset('csv', data_files=data_files)
            train_dataset = get_dataset_by_split(dataset_raw, 'train', add_prefix_token)
            val_dataset = get_dataset_by_split(dataset_raw, 'val', add_prefix_token)
            test_dataset = get_dataset_by_split(dataset_raw, 'test', add_prefix_token)
            ood_dataset = get_dataset_by_split(dataset_raw, TEST_OOD, add_prefix_token)
            benchmarks = (('ood_' + dataset_name, ood_dataset),) + benchmarks
            
        else:
            data_util = DatasetUtil(dataset_name=dataset_name, max_length=args.max_seq_length, tokenizer=tokenizer)
            dataset = data_util.get_dataset(dataset_name, split=None)

            if dataset_name == args.task_name:
                train_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][0]])
                test_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
                val_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][2]])

                # Since NewsCategory-Modified is a pure semantic shift dataset, the OOD benchmark comes from the same dataset
                if dataset_name == 'news-category-modified' or dataset_name == 'clinc150':
                    test_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][3]])
                    ood_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
                    benchmarks = (('ood_' + dataset_name, ood_dataset),) + benchmarks

            else:
                ood_dataset = process_dataset_split(dataset[data_util.dataset_to_split[dataset_name][1]])
                benchmarks = (('ood_' + dataset_name, ood_dataset),) + benchmarks

    

    tokenizer.save_pretrained(args.savedir)
    train(args, model, train_dataset, val_dataset, test_dataset, benchmarks)
    logger.info('Done.')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(exp_configs.CUDA_VISIBLE_DEVICES)
    main(exp_configs)
