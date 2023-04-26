import torch
import argparse
from fastchat.conversation import get_default_conv_template, compute_skip_echo_len
from fastchat.serve.inference import load_model
from promptGenerator import PromptGenerator


def main(args): 
    model, tokenizer = load_model(
        args.vicuna_dir,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )

    conv = get_default_conv_template(args.vicuna_dir).copy()
    if args.knowledge_base:
        prompt_generator = PromptGenerator(knowledge_dir = 'documents', k = 5, chunk_length = 128)

    while True:
        if args.knowledge_base:
            question = input("USER: ")
            msg = prompt_generator.get_prompt([question])
            # print(msg)
        else:
            question = input("USER: ")
            msg = question

        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)

        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).to(args.device),
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        
        conv.correct_message(question)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        skip_echo_len = compute_skip_echo_len(args.vicuna_dir, conv, prompt)
        outputs = outputs[skip_echo_len:]

        print(f"ASSISTANT: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vicuna-dir",
        type=str,
        default="vicuna-13b-v1.1",
        help="The path to the weights",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    parser.add_argument("--knowledge-base", action="store_true")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
