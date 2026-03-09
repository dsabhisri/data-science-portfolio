from bs4 import BeautifulSoup
import tiktoken
import os
import pandas as pd
import openai
from openai import OpenAI # Import the new OpenAI client

# --- Function: extract_text_from_html ---

def extract_text_from_html(file_path):
    """
    Extract text from an HTML file.

    Parameters:
    - file_path (str): Path to the HTML file.

    Returns:
    - Extracted text from the HTML.
    """
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 decoding fails
        with open(file_path, 'r', encoding="latin-1") as file:
            content = file.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Extract and return text from the HTML content
    return soup.get_text()

def split_text_by_tokens(text, max_tokens=300):
    """
    Split text based on token count using the tiktoken library.

    Parameters:
    - text (str): Text to be split.
    - max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    - List of text chunks, each not exceeding max_tokens.
    """

    # Get the encoder (tokenizer) for the specified model
    tokenizer = tiktoken.get_encoding('cl100k_base')

    # Split the text into chunks based on token count
    chunks = []
    current_chunk = ""
    for word in text.split():
        # Check if adding the word doesn't exceed the max_token count
        if len(tokenizer.encode(current_chunk + " " + word)) <= max_tokens:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    # Append any remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# --- Function: process_directory ---

def process_directory(directory, max_tokens=400):
    """
    Process all HTML files within a directory (and its subdirectories), extracting text and splitting based on token count.

    Parameters:
    - directory (str): The root directory to start the search for HTML files.
    - max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    - Dictionary with file paths as keys and lists of text chunks as values.
    """
    results = {}

    # Loop through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.html'):
                file_path = os.path.join(dirpath, filename)
                text = extract_text_from_html(file_path)
                chunks = split_text_by_tokens(text, max_tokens)
                results[file_path] = chunks


    return results


# --- Function: calculate_embeddings_for_dict ---

# Set an environment variable named 'API_KEY' with a placeholder for the OpenAI API key.
os.environ['API_KEY'] = 'sk-proj-R0t5EvB2oXzcdKBiWD2HvHjwwf0b1L9B9Qq7Uun8K7P971myu' + \
'6v1hwyjgDqvn1g1IxKdd2nqeFT3BlbkFJ4OKtM7LF7q-d4fiCFZJF3JN1LOYyQVSqpPvKH_kVt5539cUFNW5b7nguHjUzk8SCH8ZYi5FO4A'

def calculate_embeddings_for_dict(chunks_dict):
    """
    Calculate embeddings for a dictionary where each key is a file path and the
    corresponding value is a list of text chunks.

    Parameters:
    - chunks_dict (dict): Dictionary with file paths as keys and lists of text
      chunks as values.

    Returns:
    - DataFrame with file paths, all_chunks and their corresponding lists of
      embeddings.
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("API_KEY"))

    # Define the specific OpenAI embedding model to be used for generating embeddings.
    EMBEDDING_MODEL = "text-embedding-ada-002"
    # Define the number of text chunks to send to the OpenAI API in a single request.
    # This helps optimize API calls and stay within rate limits.
    BATCH_SIZE = 1000

    # Initialize lists to store flattened data for the DataFrame
    all_file_paths = []
    all_chunks_text = []
    all_embeddings_vectors = []
    file_counter = 0
    # Loop through each file path and its associated list of text chunks in the input
    # dictionary.
    for file_path, chunks in chunks_dict.items():
        embeddings_for_file = [] # Temporary list for embeddings of current file's chunks

        # Iterate through the chunks list in batches, defined by BATCH_SIZE.
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            # Calculate the end index for the current batch.
            batch_end = batch_start + BATCH_SIZE
            # Extract the current batch of text chunks.
            batch = chunks[batch_start:batch_end]
            # Removed: print(f"Processing embeddings for {file_path}, batch {batch_start} to {batch_end-1}")
            # Call the OpenAI API to create embeddings for the current batch of text,
            # using the specified model and API key from environment variables.
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch) # Updated API call
            # Iterate through the 'data' field in the API response, which contains the
            # embedding objects.
            for i, be in enumerate(response.data): # Access 'data' attribute
                # Assert that the index in the response matches the iteration index,
                # ensuring data integrity.
                assert i == be.index # Access 'index' attribute
            # Extract just the embedding vectors from each embedding object in the response.
            batch_embeddings = [e.embedding for e in response.data] # Access 'embedding' attribute
            # Add the embeddings from the current batch to the overall list of embeddings for
            # the file.
            embeddings_for_file.extend(batch_embeddings)

        # After processing all batches for a file, extend the global lists
        all_file_paths.extend([file_path] * len(chunks)) # Repeat file_path for each chunk
        all_chunks_text.extend(chunks) # Add all chunks for this file
        all_embeddings_vectors.extend(embeddings_for_file) # Add all embeddings for this file
        file_counter += 1
        print(f"Processed file {file_counter}: {file_path}") 
    # Return the dataframe containing file paths, all_chunks and their corresponding lists of
    # embeddings (one row per chunk).
    return pd.DataFrame({
        "file_path": all_file_paths,
        "text": all_chunks_text,
        "embeddings": all_embeddings_vectors
    })


# --- Main Execution Blocks ---

# Assuming the unzipped files are in a directory named 'documentation'
documentation_directory = 'documentation'

# 1. Process the directory to extract text and split into chunks
print(f"Processing HTML files in {documentation_directory}...")
chunks_by_file = process_directory(documentation_directory)
print("Text extraction and splitting complete.")


# 2. Calculate embeddings for the processed chunks
print("Calculating embeddings for all text chunks...")
embeddings_dataframe = calculate_embeddings_for_dict(chunks_by_file)
print("Embeddings calculation complete.")




# Install the Hugging Face Transformers library (uncomment to install in a fresh environment)
#%pip install transformers torch accelerate

# Purpose: Import classes needed to load causal language models and tokenizers from Hugging Face.
# Layman: These let us load a text-generating model and the tool that converts text to numbers.
from transformers import AutoModelForCausalLM, AutoTokenizer

# Purpose: Import PyTorch for tensor operations and device management.
# Layman: PyTorch is the library that runs the model on CPU/GPU.
import torch

# --- Choose a suitable base LLM ---
# For a beginner-friendly example, especially on free Colab tiers,
# a smaller model like GPT-2 is a good starting point.
# For production or more complex tasks, you'd consider Llama-2-7b, Mistral-7b, etc.

# Purpose: Specify which pretrained model to use.
# Layman: 'gpt2' is a small model that's quick to load and experiment with.
model_name = "gpt2" # Using GPT-2 as an example

# Purpose: Print which model is being loaded (simple runtime info).
print(f"Loading model: {model_name}")

# Purpose: Load the tokenizer for the chosen model from Hugging Face.
# Layman: Tokenizer converts text into token IDs the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer for {model_name} loaded successfully.")

# Purpose: Load the pretrained causal language model (for text generation).
# Layman: This downloads the model weights so we can generate or fine-tune text.
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Model {model_name} loaded successfully.")

# Purpose: Prefer Apple Silicon (MPS) on MacBooks with Apple Silicon (M1/M2),
# then CUDA if available, otherwise fall back to CPU.
# Layman: On Apple Air use the Metal Performance Shaders backend ('mps') when available.
if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Purpose: Move the model to the selected device so computations run there.
model.to(device)
print(f"Model moved to: {device}")

# Purpose: Print a header indicating we'll display model and tokenizer metadata.
print("\n--- Model and Tokenizer Details ---")

# Purpose: Print the Python class name of the loaded model so we know its architecture.
# Layman: Tells you which specific model implementation is in memory (e.g., GPT-style class).
print(f"Model architecture: {model.__class__.__name__}")

# Purpose: Print the tokenizer vocabulary size to know how many unique tokens it supports.
# Layman: Number of words/symbols the tokenizer can convert to IDs.
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Purpose: Print the total number of parameters of the model (size/complexity indicator).
# Layman: Rough idea of model size — larger usually means more capability but needs more resources.
print(f"Model number of parameters: {model.num_parameters()}")

# Purpose: Print a section header for the upcoming data preparation steps.
print("--- Data Preparation: Tokenizing text chunks ---")

 # 1. Extract the 'text' column from the embeddings_dataframe
 # Purpose: Pull the raw text strings into a plain Python list for tokenization.
 # Layman: Take the column named 'text' and make a simple list of sentences/paragraphs.
text_chunks = embeddings_dataframe['text'].tolist()
print(f"Extracted {len(text_chunks)} text chunks from embeddings_dataframe.")

 # 2. Ensure the pre-loaded tokenizer has a padding token set
 # Purpose: Some tokenizers don't have pad_token defined; set it so batching works.
 # Layman: If there's no special token used to fill short sequences, use the end-of-sequence token.
if tokenizer.pad_token is None:
    # Purpose: Point the pad token to the tokenizer's eos token to avoid errors during batching.
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer's pad_token set to eos_token.")

 # 3. Tokenize the extracted text chunks
 # Purpose: Convert raw text into token IDs and attention masks for model input.
 # Layman: Turn sentences into numbers the model understands, add padding to equalize lengths,
 # and return PyTorch tensors so we can feed them to a PyTorch model directly.

tokenized_dataset = tokenizer(text_chunks, padding=True, truncation=True, return_tensors="pt")

# Purpose: Confirm tokenization completed and show tensor shapes for input_ids and attention_mask.
# Layman: Shows the dimensions: (number_of_examples, sequence_length).
print("Text chunks tokenized successfully.")
print(f"Shape of input_ids: {tokenized_dataset['input_ids'].shape}")
print(f"Shape of attention_mask: {tokenized_dataset['attention_mask'].shape}")


# Purpose: import the Dataset class from the Hugging Face 'datasets' library.
# This lets us create Dataset objects from Python structures (like dicts of tensors).
from datasets import Dataset

# Purpose: log that we're starting to create the Hugging Face Dataset.
# Helps track progress when running long scripts.
print("--- Creating Hugging Face Dataset ---")

# Purpose: Build a Hugging Face Dataset from the tokenized tensors.
# This converts the tokenized 'input_ids' and 'attention_mask' PyTorch tensors
# into a Dataset object that the Trainer API and dataset utilities expect.
dataset = Dataset.from_dict({
    # Purpose: provide the input token IDs for each example.
    'input_ids': tokenized_dataset['input_ids'],
    # Purpose: provide the attention mask corresponding to the token IDs.
    'attention_mask': tokenized_dataset['attention_mask'],
})

# Purpose: confirm successful creation of the Dataset and print a message.
print("Hugging Face Dataset created successfully.")
# Purpose: display the dataset features (columns and types) to verify structure.
print(f"Dataset features: {dataset.features}")
# Purpose: display the number of examples contained in the Dataset.
print(f"Dataset size: {len(dataset)}")
# Purpose: print a short representation of the Dataset (useful for debugging).
print(dataset)

print("--- Preparing dataset for Causal Language Modeling ---")

# For Causal Language Models, the labels are typically the input_ids themselves.
# The model internally shifts the labels for next-token prediction.
def set_labels(examples):
    examples["labels"] = examples["input_ids"]
    return examples

# Apply the function to the dataset
dataset = dataset.map(set_labels, batched=True)

print("Labels column added to the dataset.")
print(f"Dataset features after adding labels: {dataset.features}")
print(f"First example labels: {dataset[0]['labels'][:10]}") # Print first 10 labels of the first example


print("--- Splitting dataset into training and validation sets ---")

# Split the dataset into training and validation sets
# A common split ratio is 80% for training and 20% for validation.
train_test_split = dataset.train_test_split(test_size=0.2)

train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print("Dataset split successfully.")


from transformers import TrainingArguments, Trainer

print("--- Reconfiguring Training Arguments for faster training ---")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    num_train_epochs=1, # Reduced to 1 for faster training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    #report_to=["tensorboard"], # Use report_to instead of logging_dir for tensorboard integration
    report_to="none", # Option 2: Disable it (simpler — you likely don't need it)
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=False,
)


print("Training arguments reconfigured successfully.")
print(f"Output directory: {training_args.output_dir}")
print(f"Number of epochs: {training_args.num_train_epochs}")

print("--- Re-initializing Hugging Face Trainer and starting fine-tuning ---")

# Re-instantiate the Trainer with the updated training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Trainer re-initialized successfully.")

# Call the train() method on the Trainer object to begin fine-tuning
trainer.train()

print("Fine-tuning process started.")

print("--- Saving fine-tuned model and tokenizer ---")

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

# Save the tokenizer
tokenizer.save_pretrained(training_args.output_dir)
print(f"Tokenizer saved to {training_args.output_dir}")



print("--- Evaluating model performance on the validation set...")

eval_results = trainer.evaluate(eval_dataset=val_dataset)

print("Evaluation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value}")
print("Evaluation complete.")


print("--- Generating Sample Q&A Prompts ---")

# 1. Define a Python list named qa_prompts
qa_prompts = [
    {
        'question': 'What is an LLMChain?',
        'expected_answer': 'An LLMChain is the simplest way to combine an LLM with a prompt template.'
    },
    {
        'question': 'What is LangChain?',
        'expected_answer': 'LangChain is a framework for developing applications powered by large language models (LLMs).'
    },
    {
        'question': 'How can I install LangChain?',
        'expected_answer': 'LangChain can be installed using pip: `pip install langchain`.'
    },
    {
        'question': 'What is a PromptTemplate in LangChain?',
        'expected_answer': 'A PromptTemplate is a reusable template for generating prompts to an LLM.'
    },
    {
        'question': 'What are Agents in LangChain?',
        'expected_answer': 'Agents involve an LLM making decisions about which Actions to take, observing an Observation, and repeating until the task is complete.'
    }
]

# 5. Print the qa_prompts list to display the generated Q&A pairs.
print("Generated Q&A Prompts:")
for item in qa_prompts:
    print(f"Question: {item['question']}")
    print(f"Expected Answer: {item['expected_answer']}")
    print("---")
print("Sample Q&A prompts generated successfully.")



print("--- Loading fine-tuned model and tokenizer for inference ---")

# 1. Import AutoModelForCausalLM and AutoTokenizer from the transformers library.
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. Load the fine-tuned model
fine_tuned_model = AutoModelForCausalLM.from_pretrained('./gpt2_finetuned')
print("Fine-tuned model loaded successfully.")

# 3. Load the corresponding tokenizer
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./gpt2_finetuned')
print("Fine-tuned tokenizer loaded successfully.")

# 4. Move the fine_tuned_model to the appropriate device
# The 'device' variable should be available from previous steps (e.g., 'cuda' or 'cpu').
fine_tuned_model.to(device)
print(f"Fine-tuned model moved to: {device}")

# 5. Set the fine_tuned_model to evaluation mode
fine_tuned_model.eval()
print("Fine-tuned model set to evaluation mode.")


print("--- Defining inference function for the fine-tuned model ---")

def generate_answer(question: str, model, tokenizer, max_length: int = 100) -> str:
    """
    Generates an answer to a given question using the fine-tuned GPT-2 model.

    Args:
        question (str): The input question.
        model: The fine-tuned causal language model.
        tokenizer: The tokenizer corresponding to the model.
        max_length (int): The maximum length of the generated answer.

    Returns:
        str: The generated answer.
    """

    # Encode the question into input IDs and move to the appropriate device
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

    # Generate a response from the model
    # Use pad_token_id to ensure consistent generation behavior when padding is needed
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]), # Max length for generated part + input part
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, # Allow for more varied responses
        top_k=50, # Consider top 50 most likely next tokens
        top_p=0.95, # Consider tokens that cumulative probability is up to 95%
        temperature=0.7 # Control randomness
    )

    # Decode the generated tokens, skipping special tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the original question from the generated text to get just the answer
    # This assumes the model completes the prompt after the question.
    if generated_text.startswith(question):
        answer = generated_text[len(question):].strip()
    else:
        answer = generated_text.strip()

    return answer

print("Inference function 'generate_answer' defined successfully.")



print("--- Processing Q&A prompts with the fine-tuned model ---")

model_answers = []

for i, qa_pair in enumerate(qa_prompts):
    question = qa_pair['question']
    expected_answer = qa_pair['expected_answer']

    print(f"\nProcessing prompt {i+1}/{len(qa_prompts)}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")

    # Generate answer using the fine-tuned model
    # max_length is set to a reasonable value for short answers, adjust if needed
    generated_answer = generate_answer(question, fine_tuned_model, fine_tuned_tokenizer, max_length=150)
    model_answers.append(generated_answer)

    print(f"Model's Answer: {generated_answer}")
    print("--------------------------------------------------")

print("Finished processing all Q&A prompts.")


print("--- Defining inference function for the fine-tuned model ---")

def generate_answer(question: str, model, tokenizer, max_length: int = 100) -> str:
    """
    Generates an answer to a given question using the fine-tuned GPT-2 model.

    Args:
        question (str): The input question.
        model: The fine-tuned causal language model.
        tokenizer: The tokenizer corresponding to the model.
        max_length (int): The maximum length of the generated answer.

    Returns:
        str: The generated answer.
    """

    # Encode the question into input IDs and move to the appropriate device
    # Explicitly include attention_mask here
    encoded_input = tokenizer(question, return_tensors='pt').to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Generate a response from the model
    # Use pad_token_id to ensure consistent generation behavior when padding is needed
    output = model.generate(
        input_ids,
        attention_mask=attention_mask, # Pass the attention mask here
        max_length=max_length + len(input_ids[0]), # Max length for generated part + input part
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, # Allow for more varied responses
        top_k=50, # Consider top 50 most likely next tokens
        top_p=0.95, # Consider tokens that cumulative probability is up to 95%
        temperature=0.7 # Control randomness
    )

    # Decode the generated tokens, skipping special tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the original question from the generated text to get just the answer
    # This assumes the model completes the prompt after the question.
    if generated_text.startswith(question):
        answer = generated_text[len(question):].strip()
    else:
        answer = generated_text.strip()

    return answer

print("Inference function 'generate_answer' defined successfully.")


print("--- Processing Q&A prompts with the fine-tuned model (re-run) ---")

model_answers = []

for i, qa_pair in enumerate(qa_prompts):
    question = qa_pair['question']
    expected_answer = qa_pair['expected_answer']

    print(f"\nProcessing prompt {i+1}/{len(qa_prompts)}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")

    # Generate answer using the fine-tuned model
    # max_length is set to a reasonable value for short answers, adjust if needed
    generated_answer = generate_answer(question, fine_tuned_model, fine_tuned_tokenizer, max_length=150)
    model_answers.append(generated_answer)

    print(f"Model's Answer: {generated_answer}")
    print("--------------------------------------------------")

print("Finished processing all Q&A prompts (re-run).")












