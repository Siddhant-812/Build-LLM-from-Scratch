import streamlit as st
import torch
import os
import requests
import math
import pandas as pd
import bitsandbytes.nn as bnb

# --- Import all our project modules ---
from model import GPTModel, GPTClassificationModel
from tokenizer import BPETokenizer
from lora import LoRALayer
from generation import generate_text
from evaluate import calculate_perplexity

# --- App Configuration ---
st.set_page_config(page_title="LLM From Scratch Showcase", layout="wide")

# --- Model & Tokenizer Configuration ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
CONTEXT_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File Paths ---
WEIGHTS_PATH = "model_weights.pth"
FINETUNED_CLASSIFIER_PATH = "finetuned_classifier_weights.pth"
LORA_CLASSIFIER_PATH = "lora_classifier_weights.pth"
QLORA_INSTRUCTION_PATH = "qlora_instruction_weights.pth"

# --- Caching Models for Performance ---
@st.cache_resource
def load_base_model_and_tokenizer():
    """Loads the base GPT model and its tokenizer."""
    text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
    tokenizer = BPETokenizer()
    tokenizer.train(text, VOCAB_SIZE)
    
    model = GPTModel(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, CONTEXT_LENGTH)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.to(DEVICE)
    return model, tokenizer

@st.cache_resource
def load_full_finetuned_classifier():
    """Loads the fully finetuned classification model."""
    base_model, _ = load_base_model_and_tokenizer()
    model = GPTClassificationModel(base_model, num_classes=2)
    if os.path.exists(FINETUNED_CLASSIFIER_PATH):
        model.load_state_dict(torch.load(FINETUNED_CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    return model

@st.cache_resource
def load_lora_classifier():
    """Loads the base model and injects the finetuned LoRA weights."""
    base_model, _ = load_base_model_and_tokenizer()
    # --- THE FIX: Check before injecting LoRA layers ---
    for block in base_model.blocks:
        if isinstance(block.attn.out_proj, torch.nn.Linear):
            block.attn.out_proj = LoRALayer(block.attn.out_proj, rank=8)
    
    model = GPTClassificationModel(base_model, num_classes=2)
    if os.path.exists(LORA_CLASSIFIER_PATH):
        model.load_state_dict(torch.load(LORA_CLASSIFIER_PATH, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    return model

@st.cache_resource
def load_qlora_instructor():
    """Loads the quantized model and injects the QLoRA instruction-tuned weights."""
    base_model, _ = load_base_model_and_tokenizer()
    
    # --- THE FIX: Check before quantizing and injecting ---
    # This prevents re-quantizing an already quantized model
    is_quantized = any(isinstance(m, bnb.Linear4bit) for m in base_model.modules())

    if not is_quantized:
        for block in base_model.blocks:
            for head in block.attn.heads:
                head.q_proj = bnb.Linear4bit(head.q_proj.in_features, head.q_proj.out_features, bias=False)
                head.k_proj = bnb.Linear4bit(head.k_proj.in_features, head.k_proj.out_features, bias=False)
                head.v_proj = bnb.Linear4bit(head.v_proj.in_features, head.v_proj.out_features, bias=False)
            block.ff.net[0] = bnb.Linear4bit(block.ff.net[0].in_features, block.ff.net[0].out_features, bias=True)
            block.ff.net[2] = bnb.Linear4bit(block.ff.net[2].in_features, block.ff.net[2].out_features, bias=True)

    for block in base_model.blocks:
        if isinstance(block.attn.out_proj, torch.nn.Linear):
            block.attn.out_proj = LoRALayer(block.attn.out_proj, rank=8)
    
    model = base_model.to(DEVICE)
    if os.path.exists(QLORA_INSTRUCTION_PATH):
        model.load_state_dict(torch.load(QLORA_INSTRUCTION_PATH, map_location=DEVICE), strict=False)
    return model

# --- Main App UI ---
st.title("Build an LLM From Scratch: Interactive Showcase")
st.markdown("This application demonstrates the models we built and finetuned throughout the project.")

with st.spinner("Loading all models, please wait..."):
    base_model, tokenizer = load_base_model_and_tokenizer()
    classifier_full = load_full_finetuned_classifier()
    classifier_lora = load_lora_classifier()
    instructor_qlora = load_qlora_instructor()
st.success("All models loaded successfully!")

# ... (The rest of the app UI code is exactly the same as before) ...
st.sidebar.title("Model Demonstrations")
page = st.sidebar.radio("Choose a model to test:", 
                        ["Text Generation (Base Model)", 
                         "Sentiment Analysis (Finetuned)", 
                         "Instruction Following (QLoRA)", 
                         "Evaluation & Efficiency"])

if page == "Text Generation (Base Model)":
    st.header("Shakespearean Text Generation")
    st.markdown("This model was pretrained on the TinyShakespeare dataset. Enter a prompt and it will generate text in a similar style.")
    prompt = st.text_input("Enter your prompt:", "To be, or not to be, that is the")
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            generated_text = generate_text(base_model, tokenizer, prompt, 150, CONTEXT_LENGTH, DEVICE, temperature=0.8, top_k=50)
            st.text_area("Generated Output:", generated_text, height=300)

elif page == "Sentiment Analysis (Finetuned)":
    st.header("Sentiment Analysis")
    st.markdown("Here we test two models finetuned for sentiment classification on Amazon reviews.")
    text_to_classify = st.text_input("Enter a sentence to classify:", "This phone is fantastic and the camera is superb!")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Full Finetuning")
        st.markdown("This model had all of its weights updated during finetuning.")
        if st.button("Classify with Full Model"):
            with torch.no_grad():
                classifier_full.eval()
                tokens = tokenizer.encode(text_to_classify)
                padded = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long, device=DEVICE)
                padded[0, :min(len(tokens), CONTEXT_LENGTH)] = torch.tensor(tokens[:CONTEXT_LENGTH])
                logits = classifier_full(padded)
                pred = torch.argmax(logits, dim=1).item()
                sentiment = "Positive" if pred == 1 else "Negative"
                st.success(f"Predicted Sentiment: **{sentiment}**")
    
    with col2:
        st.subheader("LoRA Finetuning")
        st.markdown("This model was finetuned by training only a tiny fraction (<1%) of its parameters.")
        if st.button("Classify with LoRA Model"):
            with torch.no_grad():
                classifier_lora.eval()
                tokens = tokenizer.encode(text_to_classify)
                padded = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.long, device=DEVICE)
                padded[0, :min(len(tokens), CONTEXT_LENGTH)] = torch.tensor(tokens[:CONTEXT_LENGTH])
                logits = classifier_lora(padded)
                pred = torch.argmax(logits, dim=1).item()
                sentiment = "Positive" if pred == 1 else "Negative"
                st.success(f"Predicted Sentiment: **{sentiment}**")

elif page == "Instruction Following (QLoRA)":
    st.header("Instruction Following with QLoRA")
    st.markdown("This model was finetuned on the Dolly-15k dataset using quantization and LoRA. Ask it a question!")
    st.warning("Note: The model is very small and was trained on a tiny subset of data, so its answers are not expected to be accurate, but will follow the instruction format.")
    instruction = st.text_input("Enter your instruction:", "What is the capital of France?")
    if st.button("Get Response"):
        with st.spinner("Generating response..."):
            prompt = (f"### Instruction:\n{instruction}\n\n### Context:\n\n\n### Response:\n")
            generated_text = generate_text(instructor_qlora, tokenizer, prompt, 100, CONTEXT_LENGTH, DEVICE)
            st.text_area("Model Response:", generated_text, height=300)

elif page == "Evaluation & Efficiency":
    st.header("Model Evaluation and Efficiency")
    st.subheader("Language Modeling Performance (Perplexity)")
    st.markdown("Perplexity measures how well a language model predicts a sample of text. A lower score is better.")
    with st.spinner("Calculating perplexity on a test set... (This may take a moment)"):
        test_text = requests.get("https://www.gutenberg.org/files/55/55-0.txt").text
        ppl = calculate_perplexity(base_model, test_text, tokenizer, DEVICE)
        st.metric(label="Perplexity on 'The Wizard of Oz'", value=f"{ppl:.2f}")

    st.subheader("Finetuning Efficiency: Full vs. LoRA")
    st.markdown("This chart shows the dramatic difference in the number of trainable parameters between full finetuning and LoRA.")
    total_params = sum(p.numel() for p in classifier_full.parameters())
    full_trainable = total_params
    
    lora_trainable_params = 0
    with torch.no_grad():
        lora_temp_model, _ = load_base_model_and_tokenizer()
        for p in lora_temp_model.parameters(): p.requires_grad = False
        for block in lora_temp_model.blocks:
            if isinstance(block.attn.out_proj, torch.nn.Linear):
                block.attn.out_proj = LoRALayer(block.attn.out_proj, rank=8)
        temp_classifier = GPTClassificationModel(lora_temp_model, 2)
        lora_trainable = sum(p.numel() for p in temp_classifier.parameters() if p.requires_grad)

    param_data = {'Method': ['Full Finetuning', 'LoRA Finetuning'], 'Trainable Parameters': [full_trainable, lora_trainable]}
    df = pd.DataFrame(param_data)
    st.bar_chart(df.set_index('Method'))
    st.info(f"""
    - **Total Model Parameters:** {total_params:,}
    - **Full Finetuning Parameters:** {full_trainable:,}
    - **LoRA Finetuning Parameters:** {lora_trainable:,} (**{100 * lora_trainable / total_params:.2f}%** of the total!)
    """)