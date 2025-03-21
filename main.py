import os
import torch
import torchaudio
from dotenv import load_dotenv
import argparse
from pathlib import Path
import time
import logging
from huggingface_hub import login

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CSM-TTS")

# Load environment variables
load_dotenv()

# Get environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "sesame/csm-1b")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_AUDIO_LENGTH_MS = int(os.getenv("MAX_AUDIO_LENGTH_MS", 10000))

class CSMTTSProvider:
    def __init__(self):
        self.generator = None
        self.sample_rate = None
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def initialize(self):
        """Initialize the CSM model for text-to-speech generation."""
        logger.info(f"Initializing CSM model on {DEVICE}...")
        logger.info(f"Authenticating with Hugging Face...")
        
        # Login to Hugging Face Hub
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set. You may encounter access issues for gated models.")
        else:
            login(token=HF_TOKEN)
        
        try:
            # Import and load the CSM model
            from generator import load_csm_1b
            
            self.generator = load_csm_1b(device=DEVICE)
            self.sample_rate = self.generator.sample_rate
            logger.info(f"CSM model initialized. Sample rate: {self.sample_rate}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CSM model: {str(e)}")
            return False
    
    def generate_speech(self, text, speaker=0, output_filename=None):
        """
        Generate speech from text using the CSM model.
        
        Args:
            text (str): The text to convert to speech
            speaker (int): Speaker ID (default: 0)
            output_filename (str, optional): Output file name. If None, a timestamp-based name is used.
            
        Returns:
            str: Path to the generated audio file
        """
        if self.generator is None:
            if not self.initialize():
                return None
        
        try:
            logger.info(f"Generating speech for text: '{text}'")
            
            # Generate audio using CSM
            audio = self.generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
            )
            
            # Create output filename if not provided
            if output_filename is None:
                timestamp = int(time.time())
                output_filename = f"speech_{timestamp}.wav"
            
            # Ensure output filename has .wav extension
            if not output_filename.endswith('.wav'):
                output_filename += '.wav'
            
            output_path = self.output_dir / output_filename
            
            # Save the audio file
            torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), self.sample_rate)
            logger.info(f"Audio saved to {output_path}")
            
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate speech: {str(e)}")
            return None

def main():
    """Main function to run the CSM TTS Provider from command line."""
    parser = argparse.ArgumentParser(description="Sesame CSM-1B Text-to-Speech Provider")
    parser.add_argument("--text", "-t", type=str, help="Text to convert to speech")
    parser.add_argument("--speaker", "-s", type=int, default=0, help="Speaker ID (default: 0)")
    parser.add_argument("--output", "-o", type=str, help="Output file name (optional)")
    args = parser.parse_args()
    
    # If no text provided, use a default example
    if not args.text:
        args.text = "Hello! This is Sesame CSM-1B text-to-speech model speaking."
        logger.info(f"No text provided. Using example: '{args.text}'")
    
    # Initialize the provider and generate speech
    provider = CSMTTSProvider()
    output_path = provider.generate_speech(args.text, args.speaker, args.output)
    
    if output_path:
        print(f"\nSpeech generated successfully and saved to: {output_path}")
    else:
        print("\nFailed to generate speech. Check logs for details.")

if __name__ == "__main__":
    main()