# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import csv
import os
import re
import signal
import time
import random
import argparse
from ctypes import cdll
from time import perf_counter_ns

import torch

from transformers import BloomTokenizerFast, BloomForCausalLM
from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM

RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    RUNNING = False
########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def cast(model, fp16=True):
    if fp16:
        model.half()
    return model



########################################################################
# model


def create_model():
    return BloomForCausalLM.from_pretrained("bigscience/bloom-560m")


def create_tokenizer():
    return BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


########################################################################
# sample

def sample(
    device,
    model,
    tokenizer,
    context,
    batch_size,
    input_ids_len,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length_sample=128,
    max_length=2048,
    control=False,
    priority=0
):

    # input_ids = tokenizer(
    #     context,
    #     truncation=True,
    #     padding=True,
    #     max_length=max_length,
    #     return_tensors='pt',
    # ).input_ids
    # batch_size = 8
    # input_ids_len = 512
    # max_length_sample = 2  # num of tokens to be generated from the model.
    input_ids = (torch.rand((batch_size, input_ids_len)) * 10000).long()  # .cuda()

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        # print(tokens.shape)
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def test_truncate():

    assert truncate('\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#') == '\nif len_a > len_b:\n    result = a\nelse:\n    result = b'

# https://github.com/netx-repo/PipeSwitch/blob/f321d399e501b79ad51da13074e2aecda36cb06a/pipeswitch/worker_common.py#L40
def insert_layer_level_sync(mod):
    def hook_terminate(mod, input, output):
        torch.cuda.synchronize()
        # print("added sync")
    if len(list(mod.children())) == 0:
        mod.register_forward_hook(hook_terminate)
    else:
        for child in mod.children():
            insert_layer_level_sync(child)

########################################################################
# main


def main():
    signal.signal(signal.SIGINT, signal_handler)

    # (0) constants

    models = ['bloom-560m']


    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='bloom-560m')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', type=bool, default=True)
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--no-fp16', action="store_true")
    parser.add_argument('--pad', type=int, default=50256)
    parser.add_argument('--context', type=str, default='def helloworld():')
    parser.add_argument('--output_file_path', type=str)
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--sync-level', type=str, default='kernel')
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)
    device = torch.device(args.device)

    use_fp16 = True
    if (args.no_fp16 or device.type == "cpu"):
        use_fp16 = False

    if args.model.startswith("codegen-16B"):
        use_fp16 = True

    ckpt = f'./checkpoints/{args.model}'


    # (3) load
    if args.control and args.priority > 0:
        print('load libgeek.so', flush=True)
        lib = cdll.LoadLibrary(os.path.abspath("../gpu_sched_new/gpu-sched-exp/pytcppexp/libgeek.so"))
    else:
        lib = None

    with print_time('loading parameters'):
        model = create_model().to(device)
        if args.sync_level == "layer":
            insert_layer_level_sync(model)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = args.pad

    if args.output_file_name and args.output_file_path:

        csv_filename = os.path.join(
            args.output_file_path, args.output_file_name + ".csv")
        csv_fh = open(csv_filename, 'w', 1)
        csv_writer = csv.writer(csv_fh, lineterminator='\n')
        csv_writer.writerow(
            ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
             'max_allocated_gpu_memory_allocated_byte',
             'max_reserved_gpu_memory_byte'])
    # (4) sample
    global RUNNING
    while RUNNING:
        # with print_time('sampling'):
        if lib is not None:
            try:
                suffix = os.getenv("SUFFIX", None)
                assert suffix is not None
                lib.setMem(1, suffix.encode())
            except Exception as e:
                print(e)
        torch.cuda.nvtx.range_push("bloomInference")
        start_t: int = perf_counter_ns()
        completion = sample(
            device=device, model=model, tokenizer=tokenizer,
            context=args.context,
            batch_size=args.batch_size,
            input_ids_len=512,
            pad_token_id=args.pad,
            num_return_sequences=args.num_return_sequences,
            temp=args.t, top_p=args.p, max_length_sample=args.max_length)[0]
        end_t: int = perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        if lib is not None:
            try:
                assert suffix is not None
                lib.setMem(0, suffix.encode())
            except Exception as e:
                print(e)

        if args.output_file_name and args.output_file_path:
            csv_writer.writerow([
                start_t, end_t, (end_t - start_t) / 1000000])
            csv_fh.flush()
        truncation = truncate(completion)
        # sleep(1)

        print("infer delay = ", (end_t - start_t) / 1000000)
        # print('=' * 100)
        # print(completion)
        # print('=' * 100)
        # print(args.context+truncation)
        # print('=' * 100)


if __name__ == '__main__':
    test_truncate()
    main()
    print('done.')
