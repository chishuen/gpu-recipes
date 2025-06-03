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
import datetime

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.distributed import DistributedDataParallelConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging


def setup_model_and_trainer(
    fp8: bool,
    mock_ckpt: bool,
    account_for_embedding_in_pipeline_split: bool,
    account_for_loss_in_pipeline_split: bool,
    model_name_or_path: str,
    input_sequence_length: int,
    global_batch_size: int,
    nodes: int,
    tp_size: int,
    pp_size: int,
    vpp_size: int,
    cp_size: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    tokenizer_name_or_path: str,
    scheduler,
    max_grad_norm: float,
    eval_frequency: int,
    log_frequency: int,
    max_steps: int,
    tp_comm_overlap: bool,
    gc_interval: int,
    *,
    logger,
    callbacks: list,
):
    logging.info("loading model")

    if "llama-3.1-8b" in model_name_or_path.lower():
        model_config = llm.Llama31Config8B()
        model_config.seq_length = input_sequence_length
        model = llm.LlamaModel(model_config, tokenizer=None)
    elif "llama-3.1-70b" in model_name_or_path.lower():
        model_config = llm.Llama31Config70B()
        model_config.seq_length = input_sequence_length
        model = llm.LlamaModel(model_config, tokenizer=None)
    elif "llama-3.1-405b" in model_name_or_path.lower():
        model_config = llm.Llama31Config405B()
        model_config.seq_length = input_sequence_length
        model = llm.LlamaModel(model_config, tokenizer=None)
    elif "qwen2.5-32b" in model_name_or_path.lower():
        model_config = llm.Qwen25Config32B()
        model_config.seq_length = input_sequence_length
        model = llm.Qwen2Model(model_config, tokenizer=None)
    else:
        raise ValueError(f"Unknown model specified: {model_name_or_path}")

    model.cross_entropy_loss_fusion = True
    model.gradient_accumulation_fusion = True
    model.bias_activation_fusion = True
    model.bias_dropout_add_fusion = True
    model.masked_softmax_fusion = True
    model.tp_only_amax_red = True
    model.apply_query_key_layer_scaling = True
    model.persist_layer_norm = True
    model.apply_rope_fusion = True
    model.mcore_gpt = True

    resume = None
    if not mock_ckpt:
        resume = nl.AutoResume(
            restore_config=nl.RestoreConfig(path="/ssd/checkpoints")
        )

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp_size,
        sequence_parallel=True if tp_size > 1 else False,
        use_tp_pp_dp_mapping=True,
        context_parallel_size=cp_size,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            align_param_gather=True,
            fp8_param_gather=True,
            average_in_collective=True,
            bucket_size=200,
            gradient_reduce_div_fusion=True,
        ),
        ckpt_load_strictness=False,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
    )

    if fp8:
        precision = nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            # fp8
            fp8="hybrid",
            fp8_margin=0,
            fp8_amax_history_len=1024,
            fp8_amax_compute_algo="max",
            #fp8_recipe=fp8_recipe,
            fp8_param_gather=True,
        )
    else:
        precision = nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer=optimizer_name,
        lr=learning_rate,
        weight_decay=weight_decay,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        params_dtype=torch.bfloat16,
        clip_grad=max_grad_norm,
        use_distributed_optimizer=True,
    )

    if scheduler.name == "CosineAnnealing":
        opt_sched = nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=scheduler.warmup_steps
            if "warmup_steps" in scheduler
            else None,
            max_steps=scheduler.max_steps,
            min_lr=scheduler.min_lr,
        )
    elif scheduler.name == "WarmupHoldPolicy":
        opt_sched = nl.lr_scheduler.WarmupHoldPolicyScheduler(
            warmup_steps=scheduler.warmup_steps
            if "warmup_steps" in scheduler
            else None,
            warmup_ratio=scheduler.warmup_ratio
            if "warmup_steps" not in scheduler
            else None,
            hold_steps=scheduler.hold_steps,
            max_steps=scheduler.max_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler specified: {scheduler.name}")

    from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
        userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096,
        TransformerLayerTPOverlapCfg,
        BulkOverlapCfg,
        RingExchangeOverlapCfg,
        PipelineOverlapCfg
    )
    from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
        qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
        qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
        fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
        fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
        qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
        proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
        fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
        fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
        proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
        fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    )
    callbacks.append(MegatronCommOverlapCallback(
        tp_comm_overlap=tp_comm_overlap,
        tp_comm_overlap_cfg=userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
        overlap_param_gather_with_optimizer_step=False,
        align_param_gather=True,
    ))

    from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
    callbacks.append(GarbageCollectionCallback(gc_interval, gc_interval))

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=opt_sched)
    trainer = nl.Trainer(
        devices=torch.cuda.device_count(),
        num_nodes=nodes,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        logger=logger,
        enable_progress_bar=False,
        val_check_interval=eval_frequency,
        log_every_n_steps=log_frequency,
        limit_val_batches=32,
        limit_test_batches=32,
        accumulate_grad_batches=1,
        use_distributed_sampler=False,
        callbacks=callbacks
    )

    return (
        model,
        trainer,
        opt,
        resume,
    )