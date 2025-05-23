#from openai import OpenAI
from vllm import LLM
from typing import List, Dict

# client = OpenAI()

# def get_completion(prompt: str, 
#                    model : str = "gpt-3.5-turbo-0125"):
#     completion = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
#     )
#     return completion.choices[0].message.content

def setup_llm(args):
    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed
    )

    return llm

def construct_input_prompt(
    chunk: List[Dict], 
    tokenizer,
) -> List[str]:
    prompts = [ex["input"] for ex in chunk]
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False
    ) for p in prompts]
    return prompts

if __name__ == '__main__':
    prompt = """
    Q: What is the capital of France? A: """
    completion = get_completion(prompt)
    print(completion)
