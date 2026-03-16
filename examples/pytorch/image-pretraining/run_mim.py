#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "torch>=1.5.0",
#     "torchvision>=0.6.0",
#     "datasets>=1.8.0",
# ]
# ///

import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor

import transformers
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForMaskedImageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_name: str | None = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: str | None = field(
        default=None,
        metadata={"help": "The column name of the images in the files. If not set, will try to use 'image' or 'img'."},
    )
    train_dir: str | None = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: str | None = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: float | None = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    mask_patch_size: int = field(default=32, metadata={"help": "The size of the square patches to use for masking."})
    mask_ratio: float = field(
        default=0.6,
        metadata={"help": "Percentage of patches to mask."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
                "checkpoint identifier on the hub. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name_or_path: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    image_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "The size (resolution) of each image. If not specified, will use `image_size` of the configuration."
            )
        },
    )
    patch_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration."
            )
        },
    )
    encoder_stride: int | None = field(
        default=None,
        metadata={"help": "Stride to use for the encoder."},
    )
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path or hub name of a frozen teacher model (ViTModel) for feature distillation. If None, distillation is disabled."},
    )
    lambda_distill: float = field(
        default=0.01,
        metadata={"help": "Weight of the feature distillation loss (lambda * distill_loss added to recon_loss)."},
    )
    freeze_encoder_layers: str | None = field(
        default=None,
        metadata={"help": "Comma-separated layer indices or ranges to freeze, e.g. '0,1,2' or '0-5'."},
    )
    freeze_patch_embed: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the patch embedding projection weights."},
    )


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


class MIMWithDistill(nn.Module):
    """
    Wraps a MaskedImageModeling student with an optional frozen teacher for
    feature distillation.  When teacher_model is None the module behaves
    exactly like the plain student.

    Total loss = recon_loss + lambda_distill * distill_loss

    distill_loss = MSE between the projected student CLS token (last encoder
    layer, index 0, passed through distill_proj) and the teacher CLS token
    computed on the same full (unmasked) image (CLAY-style).
    """

    def __init__(self, student_model, teacher_model=None, lambda_distill=0.01):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model          # None  →  distillation disabled
        self.lambda_distill = lambda_distill

        # CLAY-style projector: map student CLS dim → teacher CLS dim before MSE
        if teacher_model is not None:
            student_dim = student_model.config.hidden_size
            teacher_dim = teacher_model.config.hidden_size
            self.distill_proj = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.distill_proj = None

        # Per-step loss values written by forward(), read by MIMTrainer
        self._last_recon_loss: torch.Tensor | None = None
        self._last_distill_loss: torch.Tensor | None = None

    def train(self, mode=True):
        """Keep teacher permanently in eval mode regardless of Trainer calls."""
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, pixel_values, bool_masked_pos=None, **kwargs):
        # ── student forward (masked) for reconstruction loss ─────────────────
        student_out = self.student(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            **kwargs,
        )

        if self.teacher is None or bool_masked_pos is None:
            self._last_recon_loss = student_out.loss.detach() if student_out.loss is not None else None
            self._last_distill_loss = None
            return student_out

        # ── student forward (unmasked) for distillation ───────────────────────
        # Separate pass on the full image so the student CLS is computed under
        # the same conditions as the teacher (no mask tokens in the input).
        student_full_out = self.student(
            pixel_values,
            bool_masked_pos=None,
            output_hidden_states=True,
        )
        # project student CLS to teacher's feature space, then L2-normalise
        student_cls = F.normalize(
            self.distill_proj(student_full_out.hidden_states[-1][:, 0, :]), dim=-1
        )

        # ── teacher forward (no grad, full unmasked image) ───────────────────
        with torch.no_grad():
            teacher_out = self.teacher(pixel_values)
        # L2-normalise teacher CLS so MSE is scale-invariant
        teacher_cls = F.normalize(teacher_out.last_hidden_state[:, 0, :], dim=-1)

        # MSE on unit-norm vectors: range [0, 4], equivalent to 2*(1 - cosine_sim)
        distill_loss = F.mse_loss(student_cls, teacher_cls)

        # record individual losses for MIMTrainer logging
        self._last_recon_loss = student_out.loss.detach()
        self._last_distill_loss = distill_loss.detach()

        # combine losses; student_out.loss is the reconstruction L1 loss
        student_out.loss = student_out.loss + self.lambda_distill * distill_loss
        return student_out


def apply_freeze(student_model, freeze_encoder_layers: str | None, freeze_patch_embed: bool):
    """Freeze selected encoder layers and/or the patch embedding projection."""
    if freeze_patch_embed:
        for p in student_model.vit.embeddings.patch_embeddings.parameters():
            p.requires_grad_(False)
        logger.info("Frozen: patch embedding projection")

    if freeze_encoder_layers:
        indices: set[int] = set()
        for part in freeze_encoder_layers.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                indices.update(range(int(start), int(end) + 1))
            else:
                indices.add(int(part))
        for i in indices:
            for p in student_model.vit.encoder.layer[i].parameters():
                p.requires_grad_(False)
        logger.info(f"Frozen encoder layers: {sorted(indices)}")


class MIMTrainer(Trainer):
    """Trainer that additionally logs recon_loss and distill_loss separately."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recon_loss_sum = 0.0
        self._distill_loss_sum = 0.0
        self._loss_step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # accumulate individual losses (unwrap DDP if needed)
        unwrapped = model.module if hasattr(model, "module") else model
        if unwrapped.training and getattr(unwrapped, "_last_recon_loss", None) is not None:
            recon = unwrapped._last_recon_loss.float()
            distill = unwrapped._last_distill_loss.float() if unwrapped._last_distill_loss is not None else recon.new_zeros(())
            # gather across DDP ranks so logged values match the gathered total loss
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(recon, op=dist.ReduceOp.AVG)
                dist.all_reduce(distill, op=dist.ReduceOp.AVG)
            self._recon_loss_sum += recon.item()
            self._distill_loss_sum += distill.item()
            self._loss_step_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # inject per-component losses whenever the trainer logs the total loss
        if "loss" in logs and self._loss_step_count > 0:
            logs["recon_loss"] = round(self._recon_loss_sum / self._loss_step_count, 4)
            logs["distill_loss"] = round(self._distill_loss_sum / self._loss_step_count, 4)
            self._recon_loss_sum = 0.0
            self._distill_loss_sum = 0.0
            self._loss_step_count = 0
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Initialize our dataset.
    ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Create config
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name_or_path:
        config = AutoConfig.from_pretrained(model_args.config_name_or_path, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # make sure the decoder_type is "simmim" (only relevant for BEiT)
    if hasattr(config, "decoder_type"):
        config.decoder_type = "simmim"

    # adapt config
    model_args.image_size = model_args.image_size if model_args.image_size is not None else config.image_size
    model_args.patch_size = model_args.patch_size if model_args.patch_size is not None else config.patch_size
    model_args.encoder_stride = (
        model_args.encoder_stride if model_args.encoder_stride is not None else config.encoder_stride
    )

    config.update(
        {
            "image_size": model_args.image_size,
            "patch_size": model_args.patch_size,
            "encoder_stride": model_args.encoder_stride,
        }
    )

    # create image processor
    if model_args.image_processor_name:
        image_processor = AutoImageProcessor.from_pretrained(model_args.image_processor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        IMAGE_PROCESSOR_TYPES = {
            conf.model_type: image_processor_class for conf, image_processor_class in IMAGE_PROCESSOR_MAPPING.items()
        }
        image_processor = IMAGE_PROCESSOR_TYPES[model_args.model_type][-1]()

    # create model
    if model_args.model_name_or_path:
        model = AutoModelForMaskedImageModeling.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedImageModeling.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # load frozen teacher for feature distillation (optional)
    teacher_model = None
    if model_args.teacher_model_name_or_path is not None:
        teacher_model = AutoModel.from_pretrained(
            model_args.teacher_model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        logger.info(f"Loaded frozen teacher model from {model_args.teacher_model_name_or_path}")

    # freeze selected student layers before wrapping
    apply_freeze(model, model_args.freeze_encoder_layers, model_args.freeze_patch_embed)

    # wrap student (+ optional teacher) into the distillation module
    model = MIMWithDistill(model, teacher_model=teacher_model, lambda_distill=model_args.lambda_distill)

    if training_args.do_train:
        column_names = ds["train"].column_names
    else:
        column_names = ds["validation"].column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original SimMIM paper
    # source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            RandomResizedCrop(model_args.image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    # create mask generator
    mask_generator = MaskGenerator(
        input_size=model_args.image_size,
        mask_patch_size=data_args.mask_patch_size,
        model_patch_size=model_args.patch_size,
        mask_ratio=data_args.mask_ratio,
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
        which patches to mask."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        examples["mask"] = [mask_generator() for i in range(len(examples[image_column_name]))]

        return examples

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # Initialize our trainer
    trainer = MIMTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "masked-image-modeling",
        "dataset": data_args.dataset_name,
        "tags": ["masked-image-modeling"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
