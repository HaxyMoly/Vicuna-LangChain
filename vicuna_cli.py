import torch
import argparse
from fastchat.model import load_model
from fastchat.conversation import get_conv_template
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

    conv = get_conv_template("vicuna_v1.1")
    if args.knowledge_base:
        prompt_generator = PromptGenerator(knowledge_dir = 'documents', k = 5, chunk_length = 128)

    while True:
        question = input("USER: ")
        if args.knowledge_base:
            msg = prompt_generator.get_prompt([question])
            # print(msg)
        else:
            msg = question

        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)

        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).to(args.device),
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        conv.correct_message(question)
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
