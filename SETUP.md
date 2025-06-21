# Setup Instructions

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

3. Verify installation:
```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy setup complete!')"
```

## Alternative Models

For better accuracy, you can use larger models:
```bash
# Medium model (more accurate)
python -m spacy download en_core_web_md

# Large model (highest accuracy)
python -m spacy download en_core_web_lg
```
