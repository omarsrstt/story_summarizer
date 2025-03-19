import os
import re
import glob
import time
from pathlib import Path
import textwrap
import json
import argparse
from typing import List, Tuple, Dict
from tqdm import tqdm

# For local model inference
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Configuration
SUMMARY_GROUP_SIZE = 5  # Number of chapters to summarize together
NOVEL_DIR = "novels"  # Base directory for novels
NOVEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), NOVEL_DIR)
MAX_LENGTH = 1024  # Max token length for local models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

class LocalNovelSummarizer:
    def __init__(self, novel_name, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", group_size=5):
        self.novel_name = novel_name
        self.novel_path = os.path.join(NOVEL_DIR, novel_name)
        self.group_size = group_size
        # self.summaries_dir = os.path.join(self.novel_path, "summaries")
        self.summaries_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summaries", novel_name)
        self.model_name = model_name
        # self.max_new_tokens = 1024
        
        # Create summaries directory if it doesn't exist
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Initialize model
        self._load_model_and_tokenizer()
        print("Model loaded successfully!")
    
    
    def _load_model_and_tokenizer(self):
        """Load model with optimizations for faster inference."""

        # Optimize for 8-bit quantization to reduce memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move to CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def extract_chapter_number(self, filename):
        """Extract chapter number from filename."""
        match = re.search(r'(\d+)', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
    
    def get_sorted_chapters(self):
        """Get all chapter files sorted by chapter number."""
        chapter_files = glob.glob(os.path.join(self.novel_path, "*.txt"))
        chapters = [(self.extract_chapter_number(f), f) for f in chapter_files]
        return sorted(chapters)
    
    def read_chapter(self, chapter_file):
        """Read content of a chapter file."""
        with open(chapter_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def group_chapters(self, sorted_chapters):
        """Group chapters according to group size."""
        groups = []
        for i in range(0, len(sorted_chapters), self.group_size):
            group = sorted_chapters[i:i+self.group_size]
            groups.append(group)
        return groups
    
    def _generate_text(self, prompt, max_new_tokens=1024, progress_bar=None):
        """Direct text generation with optimized settings."""
        # Pre-process prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set generation parameters for speed
        with torch.no_grad():
            # Record start time
            start_time = time.time()
            
            # Generate with optimized parameters
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=True,
                num_beams=1,  # Greedy decoding for speed
                # pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Calculate time
            elapsed = time.time() - start_time
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            tokens_per_second = output_tokens / elapsed
            
            if progress_bar:
                progress_bar.set_postfix(tokens=output_tokens, time=f"{elapsed:.2f}s", speed=f"{tokens_per_second:.2f}t/s")
            else:
                print(f"Generated {output_tokens} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")
        
        # Extract generated text
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from result
        prompt_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        result = result.replace(prompt_decoded, "", 1).strip()
        
        return result
    
    def _batch_process_chunks(self, chunks, chapter_range, batch_size=2):
        """Process multiple chunks in batches to better utilize GPU."""
        # This is an optimization for GPU utilization
        # Instead of processing one chunk at a time sequentially,
        # we batch process multiple chunks to better utilize the GPU
        
        all_summaries = []
        
        # Process in batches of batch_size
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_prompts = []
            
            for j, chunk in enumerate(batch_chunks):
                chunk_index = i + j
                prompt = f"""
                <task>
                Summarize the following excerpt from {chapter_range} of the novel "{self.novel_name}".
                Focus on main plot points, character developments, and important events.
                </task>

                <text>
                {chunk}
                </text>

                <summary>
                """
                batch_prompts.append(prompt)
            
            # Tokenize all prompts in the batch
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # Generate summaries for the batch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Process each output
            for j, output in enumerate(outputs):
                result = self.tokenizer.decode(output, skip_special_tokens=True)
                prompt_decoded = self.tokenizer.decode(inputs["input_ids"][j], skip_special_tokens=True)
                result = result.replace(prompt_decoded, "", 1).strip()
                all_summaries.append(result)
        
        return all_summaries
    
    def chunk_text(self, text, max_tokens=4000):
        """Split text into chunks based on token count."""
        # Estimate token count based on words
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by sentences for more coherent chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # Get token count for this sentence
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_length + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def generate_summary_for_chunk(self, chunk, chapter_range, progress_bar=None):
        """Generate summary for a specific text chunk."""
        prompt = f"""
        <task>
        Summarize the following excerpt from {chapter_range} of the novel "{self.novel_name}".
        Focus on main plot points, character developments, and important events.
        </task>

        <text>
        {chunk}
        </text>

        <summary>
        """
        
        summary = self._generate_text(prompt, progress_bar=progress_bar)
        return summary
    
    def generate_summary(self, chapters_content, chapter_numbers):
        """Generate summary using local LLM."""
        # Create a range string for chapter numbers
        if len(chapter_numbers) == 1:
            chapter_range = f"Chapter {chapter_numbers[0]}"
        else:
            chapter_range = f"Chapters {chapter_numbers[0]}-{chapter_numbers[-1]}"
        
        # Chunk the content
        chunks = self.chunk_text(chapters_content)
        
        with tqdm(total=len(chunks), desc=f"Processing chunks for {chapter_range}") as chunk_bar:
            if len(chunks) == 1:
                return self.generate_summary_for_chunk(chunks[0], chapter_range, progress_bar=chunk_bar)
            
            # If multiple chunks, summarize each and then combine
            chunk_summaries = []
            
            # Check if we should use batched processing (GPU optimization)
            if self.device == "cuda" and len(chunks) > 1:
                # Process chunks in batches for better GPU utilization
                batch_size = min(2, len(chunks))  # Adjust batch size based on GPU memory
                chunk_bar.set_description(f"Batch processing chunks for {chapter_range}")
                
                # Use batched processing with smaller batches to avoid memory issues
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    for j, chunk in enumerate(batch_chunks):
                        summary = self.generate_summary_for_chunk(chunk, f"{chapter_range} (part {i+j+1})", progress_bar=chunk_bar)
                        chunk_summaries.append(summary)
                        chunk_bar.update(1)
            else:
                # Process chunks sequentially for CPU or when only a few chunks
                for i, chunk in enumerate(chunks):
                    chunk_bar.set_description(f"Processing chunk {i+1}/{len(chunks)} for {chapter_range}")
                    chunk_summary = self.generate_summary_for_chunk(chunk, f"{chapter_range} (part {i+1})", progress_bar=chunk_bar)
                    chunk_summaries.append(chunk_summary)
                    chunk_bar.update(1)
        
        # Combine the chunk summaries
        combined_chunk_summaries = "\n\n".join(chunk_summaries)
        
        # Create a meta-summary of all chunk summaries
        with tqdm(total=1, desc=f"Creating meta-summary for {chapter_range}") as meta_bar:
            meta_prompt = f"""
                <task>
                Create a cohesive summary of {chapter_range} from the novel "{self.novel_name}" based on these partial summaries.
                Provide a unified summary that captures the main events and character developments.
                </task>

                <summaries>
                {combined_chunk_summaries}
                </summaries>

                <unified_summary>
                """
            
            meta_summary = self._generate_text(meta_prompt, progress_bar=meta_bar)
            meta_bar.update(1)
            
        return meta_summary
    
    def generate_all_summaries(self):
        """Generate summaries for all chapter groups."""
        # Get sorted chapters
        with tqdm(total=1, desc="Finding and sorting chapters") as sort_bar:
            sorted_chapters = self.get_sorted_chapters()
            sort_bar.update(1)
            
        if not sorted_chapters:
            print(f"No chapters found for novel '{self.novel_name}'")
            return
        
        # Group chapters
        chapter_groups = self.group_chapters(sorted_chapters)
        
        # Generate summaries for each group
        all_summaries = {}
        
        with tqdm(total=len(chapter_groups), desc="Processing chapter groups") as group_bar:
            for group in chapter_groups:
                chapter_numbers = [num for num, _ in group]
                chapter_contents = []
                
                group_bar.set_description(f"Processing chapters {chapter_numbers[0]}-{chapter_numbers[-1]}")
                
                # Read all chapters in the group
                with tqdm(total=len(group), desc="Reading chapter files") as read_bar:
                    for _, chapter_file in group:
                        content = self.read_chapter(chapter_file)
                        chapter_contents.append(content)
                        read_bar.update(1)
                
                # Join all chapter contents with separators
                combined_content = "\n\n====================\n\n".join(chapter_contents)
                
                # Generate summary
                summary = self.generate_summary(combined_content, chapter_numbers)
                
                # Save summary to file
                summary_filename = f"summary_chapters_{chapter_numbers[0]}-{chapter_numbers[-1]}.txt"
                summary_path = os.path.join(self.summaries_dir, summary_filename)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # Add to all summaries dict
                all_summaries[f"{chapter_numbers[0]}-{chapter_numbers[-1]}"] = summary
                
                group_bar.update(1)
        
        # Create a master summary file with all summaries
        with tqdm(total=1, desc="Creating master summary files") as master_bar:
            master_summary_path = os.path.join(self.summaries_dir, "master_summary.txt")
            with open(master_summary_path, 'w', encoding='utf-8') as f:
                for chapter_range, summary in all_summaries.items():
                    f.write(f"SUMMARY FOR CHAPTERS {chapter_range}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(summary)
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            # Also save as JSON for programmatic access
            master_summary_json = os.path.join(self.summaries_dir, "master_summary.json")
            with open(master_summary_json, 'w', encoding='utf-8') as f:
                json.dump(all_summaries, f, indent=2)
            
            master_bar.update(1)
            
        return all_summaries
    
    def generate_novel_overview(self, all_summaries):
        """Generate an overall summary of the entire novel."""
        # Combine all summaries
        combined_summaries = ""
        for chapter_range, summary in all_summaries.items():
            combined_summaries += f"Summary for chapters {chapter_range}:\n{summary}\n\n"
        
        # Chunk the combined summaries
        chunks = self.chunk_text(combined_summaries)
        
        with tqdm(total=len(chunks) + 1, desc="Generating novel overview") as overview_bar:
            if len(chunks) == 1:
                prompt = f"""
                    <task>
                    Create an overall summary of the novel "{self.novel_name}" based on these chapter summaries.
                    Focus on:
                    1. The main plot arc from beginning to end
                    2. Key character developments 
                    3. Primary themes
                    4. Most important events and turning points
                    </task>

                    <chapter_summaries>
                    {chunks[0]}
                    </chapter_summaries>

                    <novel_overview>
                    """
                
                overview = self._generate_text(prompt, max_new_tokens=1500, progress_bar=overview_bar)
                overview_bar.update(1)
                
            else:
                # Process each chunk to extract key information
                overview_bar.set_description("Processing overview chunks")
                chunk_insights = []
                
                for i, chunk in enumerate(chunks):
                    overview_bar.set_description(f"Processing overview chunk {i+1}/{len(chunks)}")
                    chunk_prompt = f"""
                        <task>
                        Extract the key plot points, character developments, and themes from these chapter summaries of the novel "{self.novel_name}" (part {i+1}).
                        </task>

                        <summaries>
                        {chunk}
                        </summaries>

                        <key_insights>
                        """
                    
                    chunk_insight = self._generate_text(chunk_prompt, progress_bar=overview_bar)
                    chunk_insights.append(chunk_insight)
                    overview_bar.update(1)
                
                # Combine the insights into a final overview
                overview_bar.set_description("Creating final novel overview")
                combined_insights = "\n\n".join(chunk_insights)
                final_prompt = f"""
                    <task>
                    Create a cohesive overall summary of the novel "{self.novel_name}" based on these key insights from different parts of the novel.
                    Focus on:
                    1. The main plot arc from beginning to end
                    2. Key character developments
                    3. Primary themes
                    4. Most important events and turning points
                    </task>

                    <insights>
                    {combined_insights}
                    </insights>

                    <novel_overview>
                    """
                
                overview = self._generate_text(final_prompt, max_new_tokens=1500, progress_bar=overview_bar)
            
            # Save the overview
            overview_path = os.path.join(self.summaries_dir, "novel_overview.txt")
            with open(overview_path, 'w', encoding='utf-8') as f:
                f.write(overview)
            
        return overview


def list_available_novels():
    """List all available novels in the novels directory."""
    if not os.path.exists(NOVEL_DIR):
        print(f"The directory '{NOVEL_DIR}' does not exist.")
        return []
    
    novels = [d for d in os.listdir(NOVEL_DIR) if os.path.isdir(os.path.join(NOVEL_DIR, d))]
    return novels


def list_available_models():
    """List some popular lightweight models suitable for text summarization."""
    models = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct - Good balance of quality and resource usage"),
        ("TheBloke/Llama-2-7B-Chat-GGML", "Llama 2 7B (GGML) - Optimized for CPU usage"),
        ("microsoft/Phi-4-mini-instruct", "Phi-4-Mini 3.8B - Small but capable model"),
        ("microsoft/phi-2", "Phi-2 2.78B - Small but capable model"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B - Very small model"),
        ("distilbert/distilgpt2", "DistilGPT2 0.088B - Very lightweight but less capable"),
    ]
    return models


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for novel chapters using local models")
    parser.add_argument("--model", type=str, help="Model to use for summarization")
    parser.add_argument("--novel", type=str, help="Name of the novel to summarize")
    parser.add_argument("--group-size", type=int, default=SUMMARY_GROUP_SIZE, help="Number of chapters to group together")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for chunk processing (GPU optimization)")
    args = parser.parse_args()
    
    # List available novels
    print("Searching for novels ...")
    novels = list_available_novels()
    
    if not novels:
        print("No novels found. Please place your novel in a directory under 'novels/'")
        return
    
    print("Available novels:")
    for i, novel in enumerate(novels, 1):
        print(f"{i}. {novel}")
    
    # Select novel
    selected_novel = args.novel
    if not selected_novel:
        selection = input("Enter the number of the novel to summarize: ")
        try:
            idx = int(selection) - 1
            selected_novel = novels[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return
    elif selected_novel not in novels:
        print(f"Novel '{selected_novel}' not found. Available novels: {', '.join(novels)}")
        return
    
    # List available models
    available_models = list_available_models()
    
    # Select model
    selected_model = args.model
    if not selected_model:
        print("\nAvailable models:")
        for i, (model_name, description) in enumerate(available_models, 1):
            print(f"{i}. {model_name} - {description}")
        print(f"{len(available_models) + 1}. Enter custom model name")
        
        model_selection = input("\nEnter the number of the model to use: ")
        try:
            idx = int(model_selection) - 1
            if idx == len(available_models):
                selected_model = input("Enter the Hugging Face model name: ")
            else:
                selected_model = available_models[idx][0]
        except (ValueError, IndexError):
            print("Invalid selection. Using default model.")
            selected_model = available_models[0][0]
    
    # Get group size
    group_size = args.group_size
    if not args.group_size:
        group_size_input = input(f"Enter the number of chapters to group together (default: {SUMMARY_GROUP_SIZE}): ")
        group_size = int(group_size_input) if group_size_input.strip() else SUMMARY_GROUP_SIZE
    
    print(f"\nProcessing novel: {selected_novel}")
    print(f"Using model: {selected_model}")
    print(f"Group size: {group_size}")
    
    # Confirm before proceeding (models can be large)
    confirm = input("\nThis will download the model if not already present, which may use significant disk space. Continue? (y/n): ")
    if confirm.lower() not in ["y", "yes"]:
        print("Operation cancelled.")
        return
    
    # Process the novel
    summarizer = LocalNovelSummarizer(selected_novel, 
                                      selected_model, 
                                      group_size)
    all_summaries = summarizer.generate_all_summaries()
    
    if all_summaries:
        print("\nGenerating overall novel summary...")
        summarizer.generate_novel_overview(all_summaries)
    
    print("\nSummarization complete!")

if __name__ == "__main__":
    main()