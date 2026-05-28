import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Initialization & Loading ---
print("Waking the archives...")
model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./victorian_adapter"
db_folder = "./chroma_db"

# Load RAG Engine
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=db_folder, embedding_function=embeddings)

# Load Model with Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
model.config.use_cache = True

print("VictorianGPT is ready.")

# --- 2. The RAG Orchestrator ---
# --- 2. The RAG Orchestrator (Upgraded for Conversational Generation) ---
def generate_response(message, history):
    # Retrieve historical context (Reduced to k=1 for casual chat so it doesn't get overwhelmed)
    docs = vectorstore.similarity_search(message, k=1)
    context_text = "\n".join([f"Excerpt from {doc.metadata['source']}:\n{doc.page_content}" for doc in docs])
    
    # STRICT System Instructions
    system_instruction = (
        "You are a sophisticated, brooding scholar from the late 19th century. "
        "You are engaging in a direct conversation with the user. "
        "Speak in elegant, gothic prose, employing metaphors and philosophical observations."
    )
    
    # We put the rules here to force the model to behave conversationally
    formatted_prompt = f"""<|im_start|>system
{system_instruction}
<|im_end|>
<|im_start|>user
[Optional Background Knowledge for Inspiration - DO NOT QUOTE DIRECTLY]
{context_text}
[User Query]
{message}
<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.85,          # Increased slightly for more creativity
            repetition_penalty=1.15,   # Forces the model to use its own words instead of copying
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full_output.split("assistant\n")[-1].strip()
    
    # Format the sources to display in the UI (only if RAG was used)
    sources = "\n\n*Atmospheric inspiration drawn from: " + ", ".join(list(set([doc.metadata['source'] for doc in docs]))) + "*"
    return reply + sources

with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=generate_response,
        title="VictorianGPT: The Archives",
        description="Conversations with a 19th-century scholar, grounded in the texts of Dickens, Brontë, Stoker, and Wilde.",
        examples=[
            "I am feeling quite fearful of the dark tonight.",
            "What happens to a ship caught in a terrible storm?",
            "How do you fare this evening?"
        ],
        cache_examples=False # <--- THIS PREVENTS THE CRASH
    )

if __name__ == "__main__":
    # In Gradio 6.0, the css parameter moved to the launch method
    demo.launch(css=custom_css)
