import logging
import os
import random
import re
from argparse import ArgumentParser
from functools import partial
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from transformers.data.data_collator import _torch_collate_batch


def get_args():
    parser = ArgumentParser()
    
    data_args_group = parser.add_argument_group("data")
    data_args_group.add_argument("--pretrained_model_name_or_path", required=True, type=str)
    data_args_group.add_argument("--max_sequence_length", default=512, type=int)

    train_args_group = parser.add_argument_group("train")
    train_args_group.add_argument("--save_dir", required=True, type=str)
    train_args_group.add_argument("--evaluation_strategy", choices=["no", "steps", "epoch"], type=str, default="epoch")
    train_args_group.add_argument("--save_strategy", choices=["no", "epoch", "steps"], type=str, default="no")
    train_args_group.add_argument("--per_device_train_batch_size", type=int, default=32)
    train_args_group.add_argument("--per_device_eval_batch_size", type=int, default=32)
    train_args_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_args_group.add_argument("--eval_accumulation_steps", type=int, default=1)
    train_args_group.add_argument("--learning_rate", type=float, default=1e-5)
    train_args_group.add_argument("--weight_decay", type=float, default=0.01)
    train_args_group.add_argument("--max_grad_norm", type=float, default=1.0)
    train_args_group.add_argument("--num_train_epochs", type=int, default=5)
    train_args_group.add_argument("--seed", type=int, default=42)
    train_args_group.add_argument("--bf16", action="store_true")
    train_args_group.add_argument("--fp16", action="store_true")
    train_args_group.add_argument("--gradient_checkpointing", action="store_true")
    train_args_group.add_argument("--generation_max_length", default=100, type=int)
    train_args_group.add_argument("--generation_num_beams", default=1, type=int)
    train_args_group.add_argument("--standard_ratio", default=0.5, type=float, help="방언 대비 표준어를 어느 비율로 넣을지 결정합니다. 0.5의 경우 방언의 절반만큼 표준어를 사용합니다. (default: 0.5)")

    wandb_args_group = parser.add_argument_group("wandb")
    wandb_args_group.add_argument("--use_wandb", action="store_true")
    wandb_args_group.add_argument("--entity", default=None, type=str)
    wandb_args_group.add_argument("--wandb_model_name", type=str)
    wandb_args_group.add_argument("--project_name", type=str, default=None)
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_features(records, tokenizer, max_length):
    features = tokenizer(
        text=[
            standard
            for standard in records["standard_form"]
        ],
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    features["label"] = tokenizer(
        text=[dialect for dialect in records["dialect_form"]],
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return features


def batchify(list_of_samples: List[Dict[str, Union[int, List[int]]]], tokenizer):
    list_of_input_ids = [sample["input_ids"] for sample in list_of_samples]
    list_of_decoer_input_ids = [[tokenizer.bos_token_id] + sample["label"] for sample in list_of_samples]
    list_of_labels = [sample["label"] + [tokenizer.eos_token_id] for sample in list_of_samples]
    list_of_lengths = [sample["length"] for sample in list_of_samples]
    max_length = max(list_of_lengths)

    input_ids: torch.Tensor = _torch_collate_batch(list_of_input_ids, tokenizer)
    labels: torch.Tensor = _torch_collate_batch(list_of_labels, tokenizer)
    decoder_input_ids = _torch_collate_batch(list_of_decoer_input_ids, tokenizer)
    attention_mask: torch.Tensor = torch.ones((len(list_of_input_ids), max_length))
    for idx, length in enumerate(list_of_lengths):
        attention_mask[idx, length:] = 0.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def compute_rouge(eval_preds, tokenizer):
    metric = evaluate.load("rouge")
    predictions, labels = eval_preds
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=predictions, references=labels)
    return result


def main():
    args = get_args()
    seed_everything(args.seed)
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

    if args.use_wandb:
        import wandb

        wandb.init(
            entity=args.entity,
            project=args.project_name,
            name=args.wandb_model_name if args.wandb_model_name is not None else args.pretrained_model_name_or_path,
            tags=["baseline"],
            group="gyeongsang_dialect",
        )
    
    # CSV 파일 로드
    same_dataset = load_dataset('csv', data_files='/gyeongsang_dialect_csv/same_forms.csv')['train']
    diff_dataset = load_dataset('csv', data_files='/gyeongsang_dialect_csv/different_forms.csv')['train']

    # same_forms.csv에서 특정 비율만 사용
    same_dataset = same_dataset.shuffle(seed=args.seed).select(range(int(len(diff_dataset) * args.standard_ratio)))

    # 두 데이터셋 합치기
    combined_dataset = concatenate_datasets([same_dataset, diff_dataset])

    # 훈련 및 검증 세트로 분할
    dataset = combined_dataset.train_test_split(test_size=1000, seed=args.seed)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    additional_tokens_for_task = ["[dialect]"]
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens_for_task})

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = model.config.decoder_start_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    train_dataset = train_dataset.map(
        lambda records: get_features(records, tokenizer, args.max_sequence_length),
        batched=True,
        remove_columns=["standard_form", "dialect_form"],
    )
    valid_dataset = valid_dataset.map(
        lambda records: get_features(records, tokenizer, args.max_sequence_length),
        batched=True,
        remove_columns=["standard_form", "dialect_form"],
    )
    batchify_ = partial(batchify, tokenizer=tokenizer)
    compute_rouge_ = partial(compute_rouge, tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.01,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        group_by_length=True,
        length_column_name="length",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else None,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=batchify_,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge_,
    )
    trainer.train()


if __name__ == "__main__":
    main()
