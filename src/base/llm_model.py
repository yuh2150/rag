import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

model_name = "meta-llama/Llama-3.2-3B-Instruct"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # torch_dtype=torch.float16,
#     quantization_config=nf4_config,
#     device_map="auto",
#     low_cpu_mem_usage=True
# ).to(device)

def get_hf_llm(model_name=model_name , max_new_token = 512, **kwargs) :
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=nf4_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_token = 512 

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_token,
        pad_token_id=tokenizer.eos_token_id,
        # device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    gen_kwargs = {
        "temperature": 0.9
    }

    llm = HuggingFacePipeline(
        pipeline= model_pipeline,
        model_kwargs=gen_kwargs
    )
    print(1111)
    return llm

# llm = get_hf_llm()
