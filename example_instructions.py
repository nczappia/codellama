# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Command: torchrun --nproc_per_node 1 example_instructions.py --ckpt_dir ~/CodeInjection/model_files/CodeLlama-7b-Instruct --tokenizer_path ~/CodeInjection/model_files/CodeLlama-7b-Instruct/tokenizer.model 

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,# = '~/CodeInjection/model_files/CodeLlama-7b-Instruct/',
    tokenizer_path: str,# = '~/CodeInjection/model_files/CodeLlama-7b-Instruct/tokenizer.model',
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    print('Starting Generator Build')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print('Finished Generator Build\n')

    print('Model type:', type(generator))

    print('Creating Instructions')
    instructions = [
        [
            {
                "role": "user",
                "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
            }
        ],
        [
            {
                "role": "system",
                "content": "Provide answers in JavaScript",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            }
        ],
    ]
    print('Instructions Created\n')

    print('Generating Results')
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print('Results completed\n')

    print('Printing Results:')
    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
    print('Finished!')


if __name__ == "__main__":
    print('Firing main\n')
    fire.Fire(main)
