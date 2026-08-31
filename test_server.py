from vllm import LLM, SamplingParams

# Sample prompts
prompts = [
    "Hello, what is your name",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# sampling_paras = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # Create an LLM
    llm = LLM(model="/home/mani/gpu/llm_inference_project/Qwen1.5-1.8B",
              gpu_memory_utilization = 0.7, #on RTX 4060, this gives 5.6GB
              max_model_len = 2048,
              )

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts)
    print("Generated output " + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == "__main__":
    main()
