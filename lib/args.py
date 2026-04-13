"""Common CLI argument definitions for training scripts."""


def add_common_args(parser):
    """Add training arguments shared across all recipes."""
    parser.add_argument("--model_name", required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="Path to JSONL or HuggingFace dataset ID")
    parser.add_argument("--output_dir", default="/workspace/outputs")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_only_model", type=lambda s: str(s).lower() == "true",
                        default=True,
                        help="If true (default), save only the model weights "
                             "(small, but DeepSpeed full resume not possible). "
                             "Set false to save full DeepSpeed state for resume.")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Resume from output_dir's latest checkpoint. Pass 'true' "
                             "to auto-detect or a specific checkpoint dir path.")


def add_deepspeed_args(parser):
    """Add DeepSpeed-specific arguments."""
    parser.add_argument("--ds_config", default=None, help="DeepSpeed config JSON path")
    parser.add_argument("--local_rank", type=int, default=-1)
