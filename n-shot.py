from datetime import datetime
from dataclasses import dataclass
import random
import pandas as pd
from pathlib import Path
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


@dataclass
class Args:
    wiki_path: Path = "data/wiki_shuffle.txt"
    nli_path: Path = "data/nli_for_simcse.csv"
    output_root_dir: Path = "/main_data"

    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    batch_size: int = 32

    num_few_shot: int = 0
    num_premises: int = 64000

    min_premise_length: int = 4
    max_premise_length: int = 32

    device: str = "cuda:0"

    def __post_init__(self):
        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        self.date = date
        self.time = time

    @property
    def output_dir(self) -> Path:
        return self.output_root_dir / f"{self.num_few_shot}-shot" / str(self.num_premises) / self.date / self.time


def create_entailment_example(premise: str, hypothesis: str = None):
    text = f'Generate one sentence that logically entails "{premise}" in the form of a statement beginning with "Answer:". Answer: "'
    if hypothesis is not None:
        text += f'{hypothesis}".'
    return text


def create_contradiction_example(premise: str, hypothesis: str = None):
    text = f'Generate one sentence that logically contradicts "{premise}" in the form of a statement beginning with "Answer:". Answer: "'
    if hypothesis is not None:
        text += f'{hypothesis}".'
    return text


def build_fewshot_examples(fewshot_examples: list[dict[str, str]]) -> tuple[str, str]:
    entailment_examples, contradiction_examples = [], []
    for example in fewshot_examples:
        entailment_example = create_entailment_example(
            premise=example["premise"].strip(),
            hypothesis=example["entailment"].strip(),
        )
        entailment_examples.append(entailment_example)

        contradiction_example = create_contradiction_example(
            premise=example["premise"].strip(),
            hypothesis=example["contradiction"].strip(),
        )
        contradiction_examples.append(contradiction_example)

    fewshot_entailment: str = "\n".join(entailment_examples)
    fewshot_contradiction: str = "\n".join(contradiction_examples)

    return fewshot_entailment, fewshot_contradiction


def filter_by_length(
    sentences: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32,
    min_length: int = 4,
) -> list[str]:
    indices: list[list[int]] = tokenizer(sentences, add_special_tokens=False)["input_ids"]
    filtered_sentences = []
    for sentence, input_ids in zip(sentences, indices):
        if min_length <= len(input_ids) <= max_length:
            filtered_sentences.append(sentence)
    return filtered_sentences


@torch.inference_mode()
def generate_hypothesis(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    device: str = "cuda:0",
) -> list[str]:
    def collate_fn(batch: list[str]) -> BatchEncoding:
        return tokenizer(batch, padding=True, truncation=False, return_tensors="pt")

    data_loader = DataLoader(
        dataset=prompts,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    ret = []

    for batch in tqdm(data_loader, total=len(data_loader)):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(**batch.to(device), max_new_tokens=64, use_cache=True)

        prompt_length = batch.input_ids.size(1)
        outputs = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        outputs = [output.split("\n")[0] for output in outputs]
        outputs = [output[:-2] if output.endswith('".') else output for output in outputs]
        ret += outputs

    return ret


@torch.inference_mode()
def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, use_cache=True,
    )
    model = model.to(args.device).eval()

    df_nli = pd.read_csv(args.nli_path, header=None)
    df_nli.columns = ["premise", "entailment", "contradiction"]

    fewshot_examples = random.sample(df_nli.to_dict(orient="records"), k=args.num_few_shot)
    fewshot_entailment, fewshot_contradiction = build_fewshot_examples(fewshot_examples)
    (args.output_dir / "fewshot_entailment.txt").write_text(fewshot_entailment)
    (args.output_dir / "fewshot_contradiction.txt").write_text(fewshot_contradiction)

    premises: list[str] = args.wiki_path.read_text().splitlines()
    print(len(premises))
    premises = filter_by_length(
        sentences=premises,
        tokenizer=tokenizer,
        max_length=args.max_premise_length,
        min_length=args.min_premise_length,
    )
    print(len(premises))
    premises = premises[: args.num_premises]
    print(len(premises))
    entailment_prompts, contradiction_prompts = [], []
    for premise in premises:
        entailment_example = create_entailment_example(premise=premise.strip())
        entailment_prompts.append(fewshot_entailment + "\n" + entailment_example)

        contradiction_example = create_contradiction_example(premise=premise.strip())
        contradiction_prompts.append(fewshot_contradiction + "\n" + contradiction_example)

    entailments = generate_hypothesis(
        prompts=entailment_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    contradictions = generate_hypothesis(
        prompts=contradiction_prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    df = pd.DataFrame.from_dict({"sent0": premises, "sent1": entailments, "hard_neg": contradictions})
    df.to_csv(str(args.output_dir / "nli_for_simcse.csv"), index=False)


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)
