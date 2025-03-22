import os
import sys
import torch
import torchaudio
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add CSM directory to Python path
csm_path = os.path.join(os.path.dirname(__file__), 'csm')
sys.path.append(csm_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CSM-TTS")

# Load environment variables
load_dotenv()

# Set up cache location - only keep HF_HOME
HF_HOME = os.getenv("HF_HOME", "D:/Dev/Models/huggingface")
os.environ["HF_HOME"] = HF_HOME

# Set Hugging Face token for authentication to access gated models
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    logger.info("Hugging Face token configured for authentication")
else:
    logger.warning("No Hugging Face token found. You may not be able to access gated models.")
    logger.warning("Get a token at https://huggingface.co/settings/tokens and set as HF_TOKEN in .env file")

# Default values without environment variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
# Increase maximum audio length to accommodate longer text
MAX_AUDIO_LENGTH_MS = 30000  # Changed from 10000 to 30000 for longer speech

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Simple example of generating speech with CSM-1B following Hugging Face documentation."""
    try:
        logger.info("Starting text-to-speech generation process...")
        logger.info(f"Using Hugging Face cache directory: {HF_HOME}")
        
        # Import the generator module from the cloned CSM repository
        from generator import load_csm_1b
        
        logger.info(f"Loading CSM-1B model on {DEVICE}...")
        
        # Load the model (it will use HF_HOME for caching)
        generator = load_csm_1b(device=DEVICE)
        logger.info("Model loaded successfully")
        
        # Define longer text to generate with a more complex paragraph
        text = "Welcome to the world of neural text-to-speech synthesis. Modern AI models can now generate incredibly natural-sounding voices that closely mimic human speech patterns. This technology has applications in audiobooks, virtual assistants, accessibility tools, and much more. The voices can convey emotion, emphasis, and natural pacing that makes the listening experience pleasant and engaging."
        logger.info(f"Generating speech for text: '{text}'")
        
        # Generate audio with female voice (speaker=1 instead of 0)
        logger.info("Starting audio generation with female voice...")
        audio = generator.generate(
            text=text,
            speaker=1,  # Changed from 0 to 1 for female voice
            context=[],
            max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
        )
        logger.info("Audio generation completed")
        
        # Define output path
        output_path = os.path.join(OUTPUT_DIR, "audio.wav")
        
        # Save the generated audio
        logger.info(f"Saving audio to {output_path}")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        logger.info(f"Audio saved successfully to {output_path}")
        
        print(f"\nSpeech generated successfully and saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        print(f"\nFailed to generate speech: {str(e)}")

if __name__ == "__main__":
    main()