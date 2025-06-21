#!/usr/bin/env python3
"""
Setup spaCy model for Telegram Gemini Bot
This script properly installs the spaCy English model in uv environments
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_spacy_model():
    """Install spaCy model using direct download"""
    try:
        # Method 1: Try using uv to install from wheel
        logger.info("🔄 Attempting to install spaCy model via uv...")
        result = subprocess.run([
            'uv', 'add', 
            'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ spaCy model installed successfully via uv")
            return True
            
        # Method 2: Try with system Python if available
        logger.info("🔄 Trying system-wide installation...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ spaCy model installed successfully via pip")
            return True
            
        # Method 3: Direct download and install
        logger.info("🔄 Trying direct download method...")
        import urllib.request
        import tempfile
        
        wheel_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        
        with tempfile.NamedTemporaryFile(suffix='.whl', delete=False) as temp_file:
            logger.info(f"Downloading {wheel_url}...")
            urllib.request.urlretrieve(wheel_url, temp_file.name)
            
            # Try to install the downloaded wheel
            result = subprocess.run([
                'uv', 'add', temp_file.name
            ], capture_output=True, text=True)
            
            os.unlink(temp_file.name)
            
            if result.returncode == 0:
                logger.info("✅ spaCy model installed successfully via direct download")
                return True
        
        logger.error("❌ All installation methods failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error installing spaCy model: {e}")
        return False

def verify_installation():
    """Verify that the spaCy model is properly installed"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model verification successful")
        logger.info(f"   Model: {nlp.meta['name']} v{nlp.meta['version']}")
        return True
    except Exception as e:
        logger.error(f"❌ spaCy model verification failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up spaCy model for Telegram Gemini Bot")
    
    # Check if already installed
    if verify_installation():
        logger.info("🎉 spaCy model is already properly installed!")
        return
    
    # Install the model
    if install_spacy_model():
        # Verify installation
        if verify_installation():
            logger.info("🎉 spaCy model setup completed successfully!")
        else:
            logger.error("❌ Installation succeeded but verification failed")
            sys.exit(1)
    else:
        logger.error("❌ Failed to install spaCy model")
        logger.info("💡 Manual installation:")
        logger.info("   1. Activate your environment")
        logger.info("   2. Run: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
        sys.exit(1)

if __name__ == "__main__":
    main()
