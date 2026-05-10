import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
import os
import torch  # Import PyTorch for model loading/running

# Load the Gema-3B Model (Adjust path if needed)
model_name = "google/gemma-3-1b" # Or your specific model name.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device="cuda")

# Function to summarize the transcript
def summarize_transcript(transcript):
    """Summarizes a transcript using the Gema-3B model."""
    try:
        # Split the transcript into chunks
        chunks = sent_tokenize(transcript)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            prompt = f"Read the following transcript:\n{chunk}\n\nSummarize this transcript in 3-5 sentences."
            input_ids = model.encode(prompt, convert_to_ids=False)
            output = model.generate(input_ids, max_length=1024, num_return_sequences=1) # Adjust length and number of sequences as needed
            summary = output[0]['generated_text']

            summaries.append(summary)
        if summaries:  #Check if any chunks were successfully processed
            return "\n".join(summaries)
        else:
          return "No summary could be generated from the transcript."

    except Exception as e:
        return f"An error occurred during processing: {e}"


# Streamlit UI
st.title("Transcript Summarizer")

with st.sidebar:
    st.markdown("Enter your transcript here:")
    transcript_file = st.text_area("Enter Transcript:", "your_transcript.txt") # Example input field

    if not transcript_file:
        st.error("Please enter a transcript file.")
    else:
        try:
            summary = summarize_transcript(transcript_file)
            st.write("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred while summarizing the transcript: {e}")

