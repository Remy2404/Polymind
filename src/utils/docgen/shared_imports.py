"""
Shared imports and configuration for the docgen package.
"""

import logging
import os
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Suppress WeasyPrint GTK warnings on Windows
warnings.filterwarnings("ignore", message=".*libgobject.*")
warnings.filterwarnings("ignore", message=".*WeasyPrint could not import.*")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Conditionally import WeasyPrint - it might not work on Windows systems without extra libraries
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML, CSS

    WEASYPRINT_AVAILABLE = True
except ImportError as e:
    logging.warning(
        f"WeasyPrint import error: {str(e)}. PDF generation will use fallback method."
    )
except Exception as e:
    logging.warning(
        f"WeasyPrint error: {str(e)}. PDF generation will use fallback method."
    )

# Try to import a fallback PDF library (reportlab) for PDF generation
REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Preformatted,
        Image,
        ListItem,
        ListFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

    REPORTLAB_AVAILABLE = True
except ImportError:
    logging.warning(
        "ReportLab not installed. Run 'pip install reportlab' for better PDF fallback support."
    )
    # Create placeholder imports for when ReportLab is not available
    getSampleStyleSheet = None
    ParagraphStyle = None
    colors = None
    A4 = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    Preformatted = None
    Image = None
    ListItem = None
    ListFlowable = None
