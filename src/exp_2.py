from transformers import RobertaConfig, RobertaModelWithHeads, RobertaModel

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2,
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)
model_new = RobertaModel.from_pretrained(
    "roberta-base",
    config=config,
)

model.roberta