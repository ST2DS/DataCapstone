from argparse import ArgumentParser

import kss
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_args():
    parser = ArgumentParser()

    model_args_group = parser.add_argument_group("model")
    model_args_group.add_argument("--model_name_or_path", required=True, type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_path = args.model_name_or_path
    model_name = "Unknown Model"
    if "ul2" in model_path:
        model_name = "UL2"
    elif "bart" in model_path:
        model_name = "BART"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device="cuda")
    _ = model.eval()

    prompts = [
        "안녕?",
        "난 여기 사과나무 밑에 있어.",
        "넌 누구지? 정말 예쁘구나...",
        "난 여우야.",
        "이리 와 나하고 놀자. 난 정말로 슬프단다.",
        "난 너하고 놀수가 없어. 난 길들여지지 않았거든.",
        "길들인다는게 뭐지?",
        "넌 여기 사는 애가 아니구나. 넌 무얼 찾고 있니?",
        "난 친구들을 찾고 있어. 길들인다는 게 무슨 뜻이지?",
        "그건 너무나 잊혀진 일이지.",
        "그건 관계를 맺는다는 뜻이야.",
    ]
    print(f"\n{model_name} Results: ")
    with torch.no_grad():
        for prompt in prompts:
            # if model_name == "ul2":
            #     prompt = "[NLU]" + prompt
            splited_sentences = kss.split_sentences(prompt)
            generated = []
            for sentence in splited_sentences:
                tokens = tokenizer.encode(sentence, return_tensors="pt").to(
                    device="cuda", non_blocking=True
                )
                gen_tokens = model.generate(
                    tokens, max_length=256, repetition_penalty=1.5
                )
                generated.append(
                    tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
                )
            print(f"{prompt} -> {' '.join(generated)}")


if __name__ == "__main__":
    main()
