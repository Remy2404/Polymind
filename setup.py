#!/usr/bin/env python3
"""
Setup script for Telegram Gemini Bot
Handles spaCy model installation and other setup tasks
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str):
    """Run a shell command with error handling"""
    try:
        logger.info(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def setup_spacy_models():
    """Download required spaCy language models"""
    models = [
        ("en_core_web_sm", "English small model"),
        # Add more models if needed
        # ("en_core_web_md", "English medium model"),
    ]
    
    for model, description in models:
        success = run_command(
            f"python -m spacy download {model}",
            f"Downloading spaCy {description}"
        )
        if not success:
            return False
    
    return True

def verify_spacy_installation():
    """Verify spaCy models are properly installed"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy installation verified successfully")
        return True
    except Exception as e:
        logger.error(f"❌ spaCy verification failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Telegram Gemini Bot setup...")
    
    # Install spaCy models
    if not setup_spacy_models():
        logger.error("❌ Setup failed during spaCy model installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_spacy_installation():
        logger.error("❌ Setup failed during verification")
        sys.exit(1)
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("You can now run the bot with: python app.py")

if __name__ == "__main__":
    main()
