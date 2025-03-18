import os
import re
import glob
from pathlib import Path
import textwrap
import json
import argparse
from typing import List, Tuple, Dict

# For local model inference
from transformers import pipeline, BitsAndBytesConfig
import torch

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
        
        # Create summaries directory if it doesn't exist
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Initialize model
        print(f"Loading model {model_name} on {DEVICE}...")
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            # model_kwargs={"torch_dtype": torch.bfloat16},
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
        )
        print("Model loaded successfully!")
    
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
    
    def chunk_text(self, text, max_chunk_size=6000):
        """Split text into manageable chunks for the model."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Rough estimate: 1 word ~ 1.5 tokens
            word_length = len(word) * 1.5  
            if current_length + word_length > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def generate_summary_for_chunk(self, chunk, chapter_range):
        """Generate summary for a specific text chunk."""
        prompt = f"""
        Please provide a summary of the following excerpt from {chapter_range} of the novel "{self.novel_name}".
        Focus on the main plot points, character developments, and important events.
        
        Text to summarize:
        {chunk}
        
        Summary:
        """
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=MAX_LENGTH,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )[0]["generated_text"]
            
            # Extract the generated summary from the response
            # (Assuming the response contains the prompt followed by the generated text)
            summary = response.split("Summary:")[1].strip() if "Summary:" in response else response.replace(prompt, "").strip()
            
            return summary
        except Exception as e:
            print(f"Error generating summary for chunk: {e}")
            return f"Error generating summary: {str(e)}"
    
    def generate_summary(self, chapters_content, chapter_numbers):
        """Generate summary using local LLM."""
        # Create a range string for chapter numbers
        if len(chapter_numbers) == 1:
            chapter_range = f"Chapter {chapter_numbers[0]}"
        else:
            chapter_range = f"Chapters {chapter_numbers[0]}-{chapter_numbers[-1]}"
        
        # Chunk the content if it's too large
        chunks = self.chunk_text(chapters_content)
        
        if len(chunks) == 1:
            return self.generate_summary_for_chunk(chunks[0], chapter_range)
        
        # If multiple chunks, summarize each and then combine
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            chunk_summary = self.generate_summary_for_chunk(chunk, f"{chapter_range} (part {i+1})")
            chunk_summaries.append(chunk_summary)
        
        # Combine the chunk summaries
        combined_chunk_summaries = "\n\n".join(chunk_summaries)
        
        # Create a meta-summary of all chunk summaries
        meta_prompt = f"""
        Please create a cohesive summary of {chapter_range} from the novel "{self.novel_name}" based on these partial summaries:
        
        {combined_chunk_summaries}
        
        Provide a unified summary that captures the main events and character developments:
        """
        
        try:
            meta_response = self.pipeline(
                meta_prompt,
                max_new_tokens=MAX_LENGTH,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )[0]["generated_text"]
            
            meta_summary = meta_response.split("Provide a unified summary that captures the main events and character developments:")[1].strip() \
                if "Provide a unified summary that captures the main events and character developments:" in meta_response \
                else meta_response.replace(meta_prompt, "").strip()
            
            return meta_summary
        except Exception as e:
            print(f"Error generating meta-summary: {e}")
            return combined_chunk_summaries  # Fallback to combined summaries
    
    def generate_all_summaries(self):
        """Generate summaries for all chapter groups."""
        # Get sorted chapters
        sorted_chapters = self.get_sorted_chapters()
        if not sorted_chapters:
            print(f"No chapters found for novel '{self.novel_name}'")
            return
        
        # Group chapters
        chapter_groups = self.group_chapters(sorted_chapters)
        
        # Generate summaries for each group
        all_summaries = {}
        
        for group in chapter_groups:
            chapter_numbers = [num for num, _ in group]
            chapter_contents = []
            
            print(f"Processing chapters {chapter_numbers[0]}-{chapter_numbers[-1]}...")
            
            # Read all chapters in the group
            for _, chapter_file in group:
                content = self.read_chapter(chapter_file)
                chapter_contents.append(content)
            
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
            
            print(f"Summary for chapters {chapter_numbers[0]}-{chapter_numbers[-1]} saved to {summary_path}")
        
        # Create a master summary file with all summaries
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
        
        print(f"All summaries combined in {master_summary_path} and {master_summary_json}")
        return all_summaries
    
    def generate_novel_overview(self, all_summaries):
        """Generate an overall summary of the entire novel."""
        # Combine all summaries
        combined_summaries = ""
        for chapter_range, summary in all_summaries.items():
            combined_summaries += f"Summary for chapters {chapter_range}:\n{summary}\n\n"
        
        # Chunk the combined summaries if needed
        chunks = self.chunk_text(combined_summaries)
        
        if len(chunks) == 1:
            prompt = f"""
            Based on the following chapter summaries from the novel "{self.novel_name}", please create an overall summary of the entire novel.
            
            Focus on:
            1. The main plot arc from beginning to end
            2. Key character developments
            3. Primary themes
            4. Most important events and turning points
            
            Chapter summaries:
            {chunks[0]}
            
            Overall novel summary:
            """
            
            try:
                response = self.pipeline(
                    prompt,
                    max_new_tokens=MAX_LENGTH,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )[0]["generated_text"]
                
                overview = response.split("Overall novel summary:")[1].strip() if "Overall novel summary:" in response else response.replace(prompt, "").strip()
            except Exception as e:
                print(f"Error generating novel overview: {e}")
                overview = f"Error generating novel overview: {str(e)}"
        else:
            # Process each chunk to extract key information
            print("Combined summaries too large, processing in chunks...")
            chunk_insights = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Processing overview chunk {i+1}/{len(chunks)}...")
                chunk_prompt = f"""
                Based on these chapter summaries from the novel "{self.novel_name}" (part {i+1}), extract the key plot points, character developments, and themes.
                
                Summaries:
                {chunk}
                
                Key insights:
                """
                
                try:
                    chunk_response = self.pipeline(
                        chunk_prompt,
                        max_new_tokens=MAX_LENGTH,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                    )[0]["generated_text"]
                    
                    chunk_insight = chunk_response.split("Key insights:")[1].strip() if "Key insights:" in chunk_response else chunk_response.replace(chunk_prompt, "").strip()
                    chunk_insights.append(chunk_insight)
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")
                    chunk_insights.append(f"Error processing chunk {i+1}: {str(e)}")
            
            # Combine the insights into a final overview
            combined_insights = "\n\n".join(chunk_insights)
            final_prompt = f"""
            Based on these key insights from different parts of the novel "{self.novel_name}", create a cohesive overall summary of the entire novel.
            
            Focus on:
            1. The main plot arc from beginning to end
            2. Key character developments
            3. Primary themes
            4. Most important events and turning points
            
            Insights from different parts of the novel:
            {combined_insights}
            
            Overall novel summary:
            """
            
            try:
                final_response = self.pipeline(
                    final_prompt,
                    max_new_tokens=MAX_LENGTH,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )[0]["generated_text"]
                
                overview = final_response.split("Overall novel summary:")[1].strip() if "Overall novel summary:" in final_response else final_response.replace(final_prompt, "").strip()
            except Exception as e:
                print(f"Error generating final novel overview: {e}")
                overview = combined_insights  # Fallback to combined insights
        
        # Save the overview
        overview_path = os.path.join(self.summaries_dir, "novel_overview.txt")
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(overview)
        
        print(f"Novel overview saved to {overview_path}")
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
        ("distilgpt2", "DistilGPT2 - Very lightweight but less capable"),
        ("microsoft/phi-2", "Phi-2 - Small but capable model"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama - Very small model"),
    ]
    return models


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for novel chapters using local models")
    parser.add_argument("--model", type=str, help="Model to use for summarization")
    parser.add_argument("--novel", type=str, help="Name of the novel to summarize")
    parser.add_argument("--group-size", type=int, default=SUMMARY_GROUP_SIZE, help="Number of chapters to group together")
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
    summarizer = LocalNovelSummarizer(selected_novel, selected_model, group_size)
    all_summaries = summarizer.generate_all_summaries()
    
    if all_summaries:
        print("\nGenerating overall novel summary...")
        summarizer.generate_novel_overview(all_summaries)
    
    print("\nSummarization complete!")

if __name__ == "__main__":
    main()