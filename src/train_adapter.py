import torch
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import RobertaConfig, RobertaModelWithHeads, RobertaModel, AutoTokenizer
from datasets import load_dataset


dataset_name = 'imdb'
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

dataset = load_dataset("rotten_tomatoes")
# Encode the input data
dataset = dataset.map(encode_batch, batched=True)

print(dataset)

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2,
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)

# Add a new adapter
model.add_adapter(dataset_name)
# Add a matching classification head
model.add_classification_head(
    dataset_name,
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
  )
# Activate the adapter
model.train_adapter(dataset_name)
input_ids = torch.tensor([dataset['train'][0]['input_ids']])
attention_mask = torch.tensor([dataset['train'][0]['attention_mask']])

outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
hidden_states = outputs[0]
pooled_outputs = torch.mean(hidden_states, dim=1)
pooled_outputs = hidden_states[:, 0, :]

print(pooled_outputs.size())
print(isinstance(model, RobertaModelWithHeads))

# print(model.heads(, return_dict=True))


# training_args = TrainingArguments(
#     learning_rate=1e-4,
#     num_train_epochs=6,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     logging_steps=200,
#     output_dir="./training_output",
#     overwrite_output_dir=True,
#     # The next line is important to ensure the dataset labels are properly passed to the model
#     remove_unused_columns=False,
# )

# def compute_accuracy(p: EvalPrediction):
#   preds = np.argmax(p.predictions, axis=1)
#   return {"acc": (preds == p.label_ids).mean()}

# trainer = AdapterTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     compute_metrics=compute_accuracy,
# )

# trainer.train()
# trainer.evaluate()
# model.save_adapter("../models", dataset_name)