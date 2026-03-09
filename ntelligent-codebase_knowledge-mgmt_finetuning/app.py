# Import the Streamlit library to build the web application
import streamlit as st

# Import model and tokenizer loaders from Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import PyTorch for tensor operations and device management
import torch

# Import os to check if the model folder exists on disk
import os


# --- Page Configuration ---
# Set the browser tab title, icon, and layout for the Streamlit app
st.set_page_config(
    page_title="LangChain Q&A - Fine-Tuned GPT-2",
    page_icon="🤖",
    layout="centered"
)

# Display the main heading and a short description on the page
st.title("LangChain Documentation Q&A")
st.markdown("Ask questions about LangChain using a **fine-tuned GPT-2** model.")


# --- Device Selection ---
# Check which hardware accelerator is available:
# NVIDIA GPU (cuda) > Apple GPU (mps) > CPU (slowest fallback)
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# --- Load Model ---
# @st.cache_resource ensures the model is loaded only once and reused
# across user interactions, avoiding slow reloads on every question
@st.cache_resource
def load_model():
    # Path where the fine-tuned model and tokenizer were saved after training
    model_path = "gpt2_finetuned_model"

    # If the folder doesn't exist, training hasn't completed yet
    if not os.path.exists(model_path):
        return None, None

    # Load the tokenizer (converts text to numbers and back)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the fine-tuned GPT-2 model (the trained brain)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move the model to the best available device (GPU or CPU)
    model.to(device)

    # Set model to evaluation mode (disables training-specific behavior like dropout)
    model.eval()

    # GPT-2 doesn't have a padding token by default, so set it to end-of-sequence token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(question, model, tokenizer, max_length=150, temperature=0.7):
    """Generate an answer to a question using the fine-tuned GPT-2 model."""

    # Convert the question text into token IDs and attention mask,
    # then move them to the same device as the model
    encoded_input = tokenizer(question, return_tensors="pt").to(device)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    # Disable gradient calculations since we're only generating, not training
    with torch.no_grad():
        # Generate text continuation from the model
        output = model.generate(
            input_ids,
            # Tell the model which tokens are real vs padding
            attention_mask=attention_mask,
            # Total output length = answer length + question length
            max_length=max_length + len(input_ids[0]),
            # Generate one answer
            num_return_sequences=1,
            # Use end-of-sequence token for padding during generation
            pad_token_id=tokenizer.eos_token_id,
            # Allow sampling for more natural responses instead of always picking the top word
            do_sample=True,
            # Only consider the top 50 most likely next tokens at each step
            top_k=50,
            # Only consider tokens whose cumulative probability is up to 95%
            top_p=0.95,
            # Controls randomness: lower = more focused, higher = more creative
            temperature=temperature,
        )

    # Convert the generated token IDs back into readable text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # The model outputs the question + answer together,
    # so remove the question part to get just the answer
    if generated_text.startswith(question):
        answer = generated_text[len(question):].strip()
    else:
        answer = generated_text.strip()

    return answer


# --- Main App ---
# Load the fine-tuned model and tokenizer (cached after first load)
model, tokenizer = load_model()

# If the model folder doesn't exist, show an error and stop the app
if model is None:
    st.error("Fine-tuned model not found at `./gpt2_finetuned/`. Please run the training script first.")
    st.stop()

# Show a success message with the device being used
st.success(f"Model loaded successfully on **{device}**")


# --- Sidebar Settings ---
# Sidebar lets the user adjust generation parameters and see sample questions
with st.sidebar:
    st.header("Settings")

    # Slider to control how long the generated answer can be
    max_length = st.slider("Max answer length (tokens)", 50, 300, 150)

    # Slider to control how creative/random the model's responses are
    temperature = st.slider("Temperature (creativity)", 0.1, 1.5, 0.7, step=0.1)

    st.markdown("---")

    # Display clickable sample questions for easy testing
    st.markdown("**Sample Questions:**")
    sample_questions = [
        "What is an LLMChain?",
        "What is LangChain?",
        "How can I install LangChain?",
        "What is a PromptTemplate in LangChain?",
        "What are Agents in LangChain?",
    ]
    # When a sample question button is clicked, it fills the text input
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state["question"] = q


# --- Question Input ---
# Text box where the user types their question
question = st.text_input(
    "Enter your question about LangChain:",
    value=st.session_state.get("question", ""),
    placeholder="e.g., What is LangChain?"
)


# --- Generate Answer ---
# When the user clicks "Get Answer" and has typed a question
if st.button("Get Answer", type="primary") and question:
    # Show a spinner while the model generates the response
    with st.spinner("Generating answer..."):
        answer = generate_answer(question, model, tokenizer, max_length, temperature)

    # Display the generated answer
    st.markdown("### Answer")
    st.write(answer)

# If no question is entered, show a helpful hint
elif not question:
    st.info("Type a question above or click a sample question from the sidebar.")
