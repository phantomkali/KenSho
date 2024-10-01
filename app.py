import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    save_directory = 'finetuned_llama_model'
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically places model on available device (e.g., GPU)
        torch_dtype=torch.float16,
        load_in_4bit=True
    )

    # Load the LoRA adapters
    peft_model = PeftModel.from_pretrained(base_model, save_directory)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    return peft_model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.image("C:/Users/rohit/Desktop/dream/dreamsinter/dream_logo.png", use_column_width=True)  # Add your logo here
st.title("Dream Interpretation with Fine-Tuned LLaMA Model")

# Input for dream description
dream_input = st.text_area("Enter your dream:")

# Button to generate interpretation
if st.button("Interpret Dream"):
    if dream_input:
        # Explicit prompt to guide the model
        prompt = f"Interpret the following dream: {dream_input}\nInterpretation:"

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate interpretation using the model, increase `max_new_tokens` for more words
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)  # Increased from 150 to 300

        # Decode the output, skipping special tokens
        interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the interpretation starting after 'Interpretation:'
        if "Interpretation:" in interpretation:
            cleaned_interpretation = interpretation.split("Interpretation:")[-1].strip()
        else:
            cleaned_interpretation = interpretation.strip()

        # Display the cleaned interpretation without dream input
        st.write(f"**Interpretation:** {cleaned_interpretation}")
    else:
        st.write("Please enter a dream description.")
