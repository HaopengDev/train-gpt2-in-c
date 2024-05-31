import os
import sys
import torch
from contextlib import nullcontext
from torch_common import write_tokenizer, GPTConfig

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # input / output, a mistake in the origin?
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output dir to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate decay frac")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how many steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # TODO: overfit_single_batch, tensorcores, device, compile, flash,
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage 0/1/2/3")
    # python -> c bridge
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    args = parser.parse_args()

    # args checking
    assert 1 <= args.sequence_length <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}
    
    # set up DDP (distributed data parallel) for torch
    ddp_rank = 0
    ddp_local_rank = 0
    zero_stage = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = "cpu"

    # calculate gradient accumulation from the desired total batch size
    tokens_per_fwdbwd = args.batch_size * args.sequence_length * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print(f"total desired batch size: {args.total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # context manager
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext()
    torch.manual_seed(42)

    # flash attention
    FLASH = 0

    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OAI pertrained checkpoint
    if args.model[0] == 'd':
        # from scratch
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    else:
        # load GPT2 model weights
        model = GPT.from_pretained(args.model)
    model.train()
    model.to("cpu")


