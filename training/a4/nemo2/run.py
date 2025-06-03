"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import hydra
import torch

import data_utils
from logging_utils import (
    PreemptiveStop,
    MetricsLogger,
    MLPerfCallback,
)
from omegaconf import DictConfig, OmegaConf
from transformers import logging, set_seed

USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)
assert USE_CUDA

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

import torch.multiprocessing as mp
from model_utils import setup_model_and_trainer
from nemo import lightning as nl
from nemo.collections import llm

OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver(
    "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
)

mp.set_start_method("spawn", force=True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    num_nodes = int(os.environ.get("NNODES", 1))
    rank = int(os.environ.get("RANK", 0))
    logger = logging.get_logger(__name__)

    OmegaConf.resolve(config)
    set_seed(config.seed)

    if rank == 0:
        logger.info(f"\n\n************** Experiment configuration of Rank {rank} ***********")
        logger.info(OmegaConf.to_yaml(config))

    if "adam" in config.optimizer.lower():
        optimizer_name = "adam"
    else:
        raise ValueError("Unsupported optimizer for GPU run")

    data_parallel_size = world_size // (
        config.tensor_parallelism
        * config.pipeline_parallelism
        * config.context_parallelism
    )

    config.global_train_batch_size = int(
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * data_parallel_size
    )

    config.global_eval_batch_size = config.global_train_batch_size
    config.global_batch_size = config.global_train_batch_size

    metrics_logger = MetricsLogger(
        init_global_step=0,
        global_batch_size=config.global_train_batch_size,
        seq_length=config.max_length,
        target_log_ppl=config.target_eval_loss,
        train_step_time_atol=config.step_time_atol
    )
    metrics_logger.log_hyperparams(config)

    callbacks = [
        MLPerfCallback(
            global_batch_size=config.global_train_batch_size,
            micro_batch_size=config.per_device_train_batch_size,
            sequence_length=config.max_length,
            init_global_step=0,
            configs=config,
        )
    ]

    if config.nsys_profile:
        from nemo.lightning.pytorch.callbacks.nsys import NsysCallback
        callbacks.append(NsysCallback(start_step=10, end_step=15, ranks=[rank], gen_shape=False))

    if config.memory_profile:
        from callback_utils import MemoryProfileCallback
        callbacks.append(MemoryProfileCallback(
            file_prefix="memdump",
            max_entries=1000000,
            rank_0_only=True,
            start_location="init",
            end_location="train_start",
            force_oom_before_stop=False,
        ))

    callbacks.append(PreemptiveStop(stop_on_step=config.max_steps))

    model, trainer, optimizer, resume = setup_model_and_trainer(
        fp8=config.fp8,
        mock_ckpt=config.mock_ckpt,
        account_for_embedding_in_pipeline_split=config.account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=config.account_for_loss_in_pipeline_split,
        model_name_or_path=config.model.name_or_path,
        input_sequence_length=config.max_length,
        global_batch_size=config.global_train_batch_size,
        nodes=num_nodes,
        tp_size=config.tensor_parallelism,
        pp_size=config.pipeline_parallelism,
        vpp_size=config.virtual_pipeline_parallelism,
        cp_size=config.context_parallelism,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        optimizer_name=optimizer_name,
        tokenizer_name_or_path=config.model.name_or_path,
        scheduler=config.sched,
        max_grad_norm=config.max_grad_norm,
        eval_frequency=config.eval_frequency,
        log_frequency=config.log_frequency,
        max_steps=config.max_steps,
        logger=metrics_logger,
        tp_comm_overlap=config.tp_comm_overlap,
        gc_interval=config.gc_interval,
        callbacks=callbacks,
    )
    ckpt = nl.ModelCheckpoint(
        save_last=False,
        save_top_k=False,
        every_n_train_steps=0,
        always_save_context=False,
        save_context_on_train_end=False,
    )

    nemo_logger = nl.NeMoLogger(
        ckpt=ckpt,
        name=config.exp_name,
        tensorboard=None,
        wandb=None,
        log_dir="/results",
    )

    if config.mock_data:
      dataset = data_utils.get_dataset_mock(config)
    else:
      dataset = data_utils.get_dataset_c4(config)
    
    if rank == 0:
      logger.info(f"\n\n************** chishuen: Model info before training ***********")
      num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      num_total_params = sum(p.numel() for p in model.parameters())
      logger.info(f"\nTotal parameters: {num_total_params:,}")
      logger.info(f"Trainable parameters: {num_trainable_params:,}")
      try:
        logger.info(f"Model:\n{model.print()}")
      except RuntimeError:
        logger.info("model.print() failed")
      logger.info("*"*40)

    llm.train(
        model=model,
        data=dataset,
        trainer=trainer,
        tokenizer="data",
        optim=optimizer,
        log=nemo_logger,
        # log=None,
        resume=resume,
    )

    if rank == 0:
      logger.info(f"\n\n************** chishuen: Model info after training ***********")
      num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      num_total_params = sum(p.numel() for p in model.parameters())
      logger.info(f"\nTotal parameters: {num_total_params:,}")
      logger.info(f"Trainable parameters: {num_trainable_params:,}")
      try:
        logger.info(f"Model:\n{model.print()}")
      except:
        logger.info("model.print() failed")
      logger.info("*"*40)


if __name__ == "__main__":
    main()