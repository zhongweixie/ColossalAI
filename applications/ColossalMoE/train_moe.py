import argparse
import os
import json

import torch
from colossal_moe.models.mixtral_checkpoint import MixtralMoECheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from colossal_moe.utils import load_checkpoint, load_model, move_to_cuda, save_checkpoint
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.moe import MOE_MANAGER, apply_load_balance
from colossalai.moe.layers import apply_load_balance
from colossalai.moe.manager import MOE_MANAGER
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from torch.utils.tensorboard import SummaryWriter

from colossal_moe.dataset.loader import (
    load_tokenized_dataset,
    DataCollatorForSupervisedDataset,   
)

@torch.no_grad()
def get_global_loss(loss, booster):
    global_loss = loss.clone().detach()
    dist.all_reduce(tensor=global_loss, op=dist.ReduceOp.SUM, group=booster.plugin.dp_group)
    global_loss.div_(booster.plugin.dp_size)
    return global_loss

def parse_args():
    # basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Parallel methods.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="The path of your saved model after finetuning.",
    )
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2,
        help=" The interval (steps) of saving checkpoints.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="The mixed precision training.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint")

    # optim
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # zero stage for all plugins
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage.")
    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size for hybrid plugin")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size for hybrid plugin")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size for hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="Microbatch size in pipeline for hybrid plugin")
    parser.add_argument("--tensorboard_dir", type=str, default="logs_dir", help="Tensorboard directory")
    parser.add_argument("--config_file", type=str, default="config_file", help="Config file")

    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations. Skip if not installed.",
    )
    parser.add_argument(
        "--use_layernorm_kernel",
        action="store_true",
        help="Use layernorm kernel. Need to install apex. Raise error if not installed.",
    )

    # load balance
    parser.add_argument(
        "--load_balance", action="store_true", help="Expert load balance. Defaults to False. Recommend to enable."
    )
    parser.add_argument("--load_balance_interval", type=int, default=1000, help="Expert load balance interval.")
    # communicate overlap
    parser.add_argument(
        "--comm_overlap",
        action="store_true",
        help="Use communication overlap for MoE. Recommended to enable for muiti-node training.",
    )
    # hierarchical all-to-all
    parser.add_argument(
        "--hierarchical_alltoall",
        action="store_true",
        help="Use hierarchical all-to-all for MoE. Recommended to enable for muiti-node training.",
    )
    
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    
    parser.add_argument("--dataset", nargs="+", default=[])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config_file, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Set plugin
    booster_kwargs = {}
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": MixtralForCausalLMPolicy(),
        "enable_fused_normalization": args.use_layernorm_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": args.precision,
        "zero_stage": args.zero_stage,
        "checkpoint_io": MixtralMoECheckpointIO,
    }
    mgr_dict = {}
    if args.plugin == "hybrid":
        plugin = MoeHybridParallelPlugin(
            pp_size=args.pp_size,
            microbatch_size=args.microbatch_size,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=args.dp_size,
            fixed_ep_size=args.ep_size,
            fixed_pp_size=args.pp_size,
            **mgr_dict,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    # Build Mixtral model
    config = MixtralConfig.from_pretrained(args.model_name)
    config.use_cache = False
    config.num_local_experts = 1
    config._attn_implementation = "flash_attention_2"
    torch.set_default_dtype(torch.bfloat16)
    model = MixtralForCausalLM(config)
    torch.set_default_dtype(torch.float32)
    model.num_experts = 8
    model = model.to(torch.bfloat16) if args.precision == "bf16" else model.to(torch.float16)
    model = model.to(get_current_device())
    replace_moe_layer(model, enable_kernel=args.use_kernel)
    coordinator.print_on_master(f"Finish init model with config:\n{config}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=4096)
    dataset = load_tokenized_dataset(dataset_paths=args.dataset, mode="train")
    dataloader = plugin.prepare_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=data_collator
    )

    # Set optimizer
    optimizer = HybridAdam(
        model_params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=args.num_epoch * len(dataloader),
        warmup_steps=args.warmup_steps
        if args.warmup_steps is not None
        else int(args.num_epoch * len(dataloader) * 0.025),
        eta_min=0.1 * args.lr,
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader)
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    coordinator.print_on_master(f"Finish init booster")
    
    if is_pp_last_stage and coordinator._local_rank == '0':
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    
    coordinator.print_on_master(f"Tensorboard logs will be saved at: {args.tensorboard_dir}")
    coordinator.print_on_master(f"Configuration file will be saved at: {args.config_file}")

    # Load ckpt
    # Load ckpt
    if args.load_checkpoint is None:
        load_model(args.model_name, model, booster, optimizer)
        coordinator.print_on_master(f"Finish load checkpoint")
    else:
        load_checkpoint(args.load_checkpoint, booster, model, optimizer, lr_scheduler)
        coordinator.print_on_master(f"Finish load optimizer")

    # Start finetuning
    coordinator.print_on_master(f"Start finetuning")
    
    model.train()
    train_dataloader_iter = iter(dataloader)
    for epoch in range(args.num_epoch):
        with tqdm(
            range(len(dataloader)),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master() if use_pipeline == False else not (is_pp_last_stage and coordinator._local_rank == '0'),
            total=len(dataloader)
        ) as pbar:
            for step in pbar:
                if use_pipeline:
                    outputs = booster.execute_pipeline(
                        train_dataloader_iter,
                        model,
                        lambda x, y: x.loss,
                        optimizer,
                        return_loss=True,
                        return_outputs=True,
                    )
                    # Backward and optimize
                    if is_pp_last_stage:
                        loss = outputs["loss"]         
                        # aux_loss = outputs["outputs"]["aux_loss"]
                        
                        global_loss = get_global_loss(loss, booster)
                        # global_aux_loss = get_global_loss(aux_loss, booster)
                        if coordinator._local_rank == "0":
                            pbar.set_postfix({"Loss": global_loss.item()})
                            writer.add_scalar(tag="Loss", scalar_value=global_loss.item(), global_step=step)
                            # writer.add_scalar(tag="Aux Loss", scalar_value=global_aux_loss.item(), global_step=step)
                            writer.add_scalar(
                                tag="Learning Rate",
                                scalar_value=lr_scheduler.get_last_lr()[0],
                                global_step=step,
                            )

                        # import torch.distributed as dist
                        # new_loss = loss.clone().detach()
                        # new_aux_loss = aux_loss.clone().detach()
                        # dist.all_reduce(tensor=new_loss, op=dist.ReduceOp.SUM, group=booster.plugin.dp_group)
                        # dist.all_reduce(tensor=new_aux_loss, op=dist.ReduceOp.SUM, group=booster.plugin.dp_group)
                        # new_loss.div_(booster.plugin.dp_size)
                        # new_aux_loss.div_(booster.plugin.dp_size)
                        # if coordinator._local_rank == '0':
                        #     pbar.set_postfix({"Loss": new_loss.item()})
                        #     writer.add_scalar(tag="Loss", scalar_value=new_loss.item(), global_step=step)
                        #     writer.add_scalar(tag="Aux Loss", scalar_value=new_aux_loss.item(), global_step=step)
                        #     writer.add_scalar(
                        #         tag="Learning Rate",
                        #         scalar_value=lr_scheduler.get_last_lr()[0],
                        #         global_step=step,
                        #     )
                else:
                    # Forward pass
                    data = next(train_dataloader_iter)
                    data = move_to_cuda(data, torch.cuda.current_device())
                    outputs = model(**data)
                    loss = outputs["loss"]
                    # Backward
                    booster.backward(loss, optimizer)
                    pbar.set_postfix({"Loss": loss.item()})
                    writer.add_scalar(tag="Loss", scalar_value=loss.item(), global_step=step)
                    writer.add_scalar(
                        tag="Learning Rate",
                        scalar_value=lr_scheduler.get_last_lr()[0],
                        global_step=step,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Apply load balance
                if (
                    args.load_balance
                    and args.load_balance_interval > 0
                    and (step + 1) % args.load_balance_interval == 0
                ):
                    coordinator.print_on_master(f"Apply load balance")
                    apply_load_balance(model, optimizer)
                # save ckeckpoint
                if (step + 1) % args.save_interval == 0:
                        coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")
                        save_checkpoint(
                            args.output_path,
                            booster,
                            model,
                            optimizer,
                            lr_scheduler,
                            epoch,
                            step,
                            args.batch_size,
                            coordinator,
                        )
                    


        # save checkpoint at the end of each epochs
        booster.save_model(model, args.output_path, shard=True, size_per_shard=5120)
        coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")

    # Finish training
    coordinator.print_on_master(f"Finish training")


if __name__ == "__main__":
    main()
