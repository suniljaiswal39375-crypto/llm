"""CLI entrypoint for pre-training / mid-training NovaLM from YAML config + dot-overrides."""

from __future__ import annotations

import argparse

from nova_llm.config import TrainConfig, load_config
from nova_llm.data import PackedShardLoader, TokenShardReader
from nova_llm.models import NovaLM
from nova_llm.training import PreTrainer, setup_distributed
from nova_llm.training.distributed import cleanup_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train NovaLM with MuonClip/AdamW & WSD/Cosine")
    parser.add_argument("--config", type=str, default="configs/train/pretrain_muonclip.yaml")
    parser.add_argument("--train-shard", type=str, required=True, help="Shard prefix (without .tokens.bin)")
    parser.add_argument("--val-shard", type=str, default=None, help="Optional validation shard prefix")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dot-notation config overrides, e.g. --override optim.lr=1e-3",
    )
    args = parser.parse_args()

    cfg: TrainConfig = load_config(TrainConfig, path=args.config, overrides=args.override)
    dist_ctx = setup_distributed(cfg.distributed)
    try:
        train_reader = TokenShardReader(args.train_shard)
        train_loader = PackedShardLoader(
            train_reader,
            batch_size=cfg.micro_batch_size,
            seq_len=cfg.seq_len,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            seed=cfg.seed,
        )
        val_loader = None
        bpt = train_reader.bytes_per_token
        if args.val_shard:
            val_reader = TokenShardReader(args.val_shard)
            val_loader = PackedShardLoader(
                val_reader,
                batch_size=cfg.micro_batch_size,
                seq_len=cfg.seq_len,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                seed=cfg.seed + 999,
                shuffle_windows=False,
            )
            bpt = val_reader.bytes_per_token

        model = NovaLM(cfg.model)
        if dist_ctx.is_master:
            print("Model parameters:", model.count_parameters())

        trainer = PreTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            bytes_per_token=bpt,
            device=args.device,
            dist_ctx=dist_ctx,
        )
        if args.resume:
            trainer.resume(args.resume)
        trainer.fit()
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
