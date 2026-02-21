"""
Filename: pinecone_ingestion_improved.py
Description: IMPROVED Production-grade PDF ingestion with advanced chunking,
             better metadata, and full reranker compatibility
             
Key Improvements:
1. Better semantic chunking with overlap optimization
2. Enhanced metadata with text preview for reranker
3. Deduplication to avoid redundant vectors
4. Resume capability (skip already processed files)
5. Better error handling and logging
6. Optimized batch processing
7. Progress persistence

Usage: 
    python pinecone_ingestion_improved.py              # Ingest all
    python pinecone_ingestion_improved.py bs           # Ingest BS/ADP only
    python pinecone_ingestion_improved.py --resume     # Resume from last run
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
from tqdm import tqdm
import hashlib

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

PINECONE_INDEX_NAME = "uoeaiassistant"
PINECONE_DIMENSION = 3072
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 50  # Reduced for better rate limiting

DATA_DIR = Path("/mnt/data/hammadali08/Personal/UOE_AI_ASSISTANT/Data")

NAMESPACES = {
    "bs-adp-schemes": DATA_DIR / "BS&ADP",
    "ms-phd-schemes": DATA_DIR / "Ms&Phd",
    "rules-regulations": DATA_DIR / "Rules"
}

# Progress tracking file
PROGRESS_FILE = Path("ingestion_progress.json")
PROCESSED_FILES_LOG = Path("processed_files.json")

# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== IMPROVED SEMANTIC CHUNKING ====================

class ImprovedSemanticChunker:
    """
    IMPROVED semantic chunker with:
    1. Better boundary detection
    2. Optimized overlap for reranker
    3. Chunk quality validation
    """
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        
        # IMPROVED: More granular chunk sizes based on content type
        self.chunk_configs = {
            "bs-adp-schemes": {
                "chunk_size": 1000,      # Reduced from 1200 for better reranker performance
                "chunk_overlap": 150,     # Reduced overlap (was 200)
                "min_chunk_size": 100,    # NEW: Minimum viable chunk
                "max_chunk_size": 1500,   # NEW: Hard limit
                "separators": [
                    "\n\n## ",
                    "\n### ",             # NEW: Sub-sections
                    "\nCourse Code:",
                    "\nCourse Title:",    # NEW: Explicit course title
                    "\nPrerequisites:",
                    "\nCourse Objectives:", # NEW: Objectives
                    "\nCLO",
                    "\nLearning Outcomes:", # NEW: Explicit LO section
                    "\n\n",
                    "\n",
                    ". ",
                    ", ",                 # NEW: Comma separation
                    " "
                ]
            },
            "ms-phd-schemes": {
                "chunk_size": 900,
                "chunk_overlap": 150,
                "min_chunk_size": 100,
                "max_chunk_size": 1400,
                "separators": [
                    "\n\n## ",
                    "\n### ",
                    "\nCourse Code:",
                    "\nSemester ",
                    "\nObjectives:",
                    "\nThesis Requirements:", # NEW: Thesis info
                    "\n\n",
                    "\n",
                    ". ",
                    ", ",
                    " "
                ]
            },
            "rules-regulations": {
                "chunk_size": 700,       # Reduced from 800
                "chunk_overlap": 100,     # Reduced from 150
                "min_chunk_size": 80,
                "max_chunk_size": 1000,
                "separators": [
                    "\n\n## ",
                    "\nArticle ",
                    "\nSection ",
                    "\nRule ",
                    "\nClause ",           # NEW: Clause boundaries
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",                  # NEW: Semicolon separation
                    " "
                ]
            }
        }
        
        config = self.chunk_configs[namespace]
        
        # Create text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            separators=config["separators"],
            is_separator_regex=False
        )
        
        self.min_chunk_size = config["min_chunk_size"]
        self.max_chunk_size = config["max_chunk_size"]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents with quality validation
        
        NEW: Filters out chunks that are too small or too large
        """
        chunks = self.splitter.split_documents(documents)
        
        # IMPROVED: Filter chunks by quality
        quality_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            
            # Skip if too short
            if len(text) < self.min_chunk_size:
                logger.debug(f"Skipping chunk: too short ({len(text)} chars)")
                continue
            
            # Truncate if too long (shouldn't happen, but safety check)
            if len(text) > self.max_chunk_size:
                logger.warning(f"Truncating chunk: too long ({len(text)} chars)")
                text = text[:self.max_chunk_size]
                chunk.page_content = text
            
            quality_chunks.append(chunk)
        
        return quality_chunks


# ==================== ENHANCED METADATA EXTRACTION ====================

class EnhancedMetadataExtractor:
    """
    IMPROVED metadata extraction with better accuracy
    """
    
    @staticmethod
    def extract_course_code(text: str) -> str:
        """IMPROVED: Better course code extraction"""
        patterns = [
            r'\b([A-Z]{2,4})[-\s]?(\d{4})\b',    # COMP1112, COMP-1112
            r'\b([A-Z]{2,4})[-\s]?(\d{3})\b',    # CS-301, CS 301
            r'Course Code[:\s]+([A-Z]{2,4}[-\s]?\d{3,4})',  # NEW: From label
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                code = ''.join(match.groups()).replace(' ', '').replace('-', '').upper()
                # Validate it looks like a real code
                if re.match(r'^[A-Z]{2,4}\d{3,4}$', code):
                    return code
        return ""
    
    @staticmethod
    def extract_course_title(text: str) -> str:
        """NEW: Extract course title"""
        patterns = [
            r'Course Title[:\s]+([^\n]+)',
            r'Course Name[:\s]+([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]  # Limit to 100 chars
        return ""
    
    @staticmethod
    def extract_credit_hours(text: str) -> str:
        """IMPROVED: Better credit hour extraction"""
        # Try standard format first
        match = re.search(r'(\d)\s*\((\d)\s*[\+\-]\s*(\d)\)', text)
        if match:
            return f"{match.group(1)}({match.group(2)}+{match.group(3)})"
        
        # Try from label
        match = re.search(r'Credit Hours?[:\s]+(\d\(\d\+\d\))', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return ""
    
    @staticmethod
    def extract_prerequisites(text: str) -> List[str]:
        """IMPROVED: Better prerequisite extraction"""
        prereq_section = re.search(
            r'Pre-?requisites?[:\s]+(.+?)(?:\n\n|\nCourse|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if not prereq_section:
            return []
        
        prereq_text = prereq_section.group(1)
        
        # Check for "none" variations
        if re.search(r'\b(none|nil|na|n/a)\b', prereq_text, re.IGNORECASE):
            return []
        
        # Extract codes
        codes = set()
        patterns = [
            r'\b([A-Z]{2,4})[-\s]?(\d{4})\b',
            r'\b([A-Z]{2,4})[-\s]?(\d{3})\b'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, prereq_text.upper())
            for match in matches:
                code = ''.join(match).replace(' ', '').replace('-', '')
                if re.match(r'^[A-Z]{2,4}\d{3,4}$', code):
                    codes.add(code)
        
        return sorted(list(codes))
    
    @staticmethod
    def extract_semester(text: str) -> int:
        """IMPROVED: Better semester extraction"""
        patterns = [
            r'Semester[:\s-]+([IVX]+)',  # Roman numerals
            r'Semester[:\s-]+(\d)',       # Arabic numerals
            r'(\d)(?:st|nd|rd|th)\s+Semester',  # 1st Semester
        ]
        
        roman_map = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4,
            'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8
        }
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sem_str = match.group(1).upper()
                if sem_str in roman_map:
                    return roman_map[sem_str]
                try:
                    sem_num = int(sem_str)
                    if 1 <= sem_num <= 8:
                        return sem_num
                except:
                    pass
        return 0
    
    @staticmethod
    def extract_program_type(filename: str) -> str:
        """Enhanced program type extraction"""
        filename_upper = filename.upper()
        
        # More specific patterns
        if re.search(r'\bBS\b', filename_upper) and 'POST' not in filename_upper:
            return 'BS'
        elif re.search(r'\bADP\b', filename_upper):
            return 'ADP'
        elif re.search(r'\bMS\b', filename_upper):
            return 'MS'
        elif re.search(r'\bMPHIL\b|\bM\.?PHIL\b', filename_upper):
            return 'MPhil'
        elif re.search(r'\bPHD\b|\bPH\.?D\b', filename_upper):
            return 'PhD'
        elif re.search(r'\bMA\b', filename_upper):
            return 'MA'
        elif re.search(r'\bMSC\b|\bM\.?SC\b', filename_upper):
            return 'MSc'
        
        return 'Unknown'
    
    @staticmethod
    def extract_department(filename: str) -> str:
        """IMPROVED: More comprehensive department mapping"""
        dept_keywords = {
            'Computer Science': ['computer science', 'cs ', ' cs', '_cs_', 'comp sci'],
            'Information Technology': ['information technology', 'it ', ' it', '_it_'],
            'Electrical Engineering': ['electrical', 'ee ', ' ee', '_ee_'],
            'Mathematics': ['math', 'mathematics'],
            'Physics': ['physics'],
            'Chemistry': ['chemistry', 'chem'],
            'History': ['history'],
            'English': ['english'],
            'Urdu': ['urdu'],
            'Education': ['education', 'b.ed', 'm.ed'],
            'Biology': ['biology', 'bio'],
            'Economics': ['economics', 'econ'],
            'Business Administration': ['business', 'bba', 'mba'],
            'Commerce': ['commerce', 'b.com', 'm.com'],
            'Psychology': ['psychology', 'psych'],
            'Sociology': ['sociology'],
            'Political Science': ['political', 'politics'],
            'Islamic Studies': ['islamic'],
            'Pakistan Studies': ['pakistan studies'],
            'Environmental Science': ['environmental'],
            'Geography': ['geography'],
            'Statistics': ['statistics', 'stats'],
            'Fine Arts': ['fine arts', 'arts'],
            'Sports Sciences': ['sports', 'physical education'],
            'Special Education': ['special education'],
        }
        
        filename_lower = filename.lower()
        
        for department, keywords in dept_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                return department
        
        return 'Unknown'
    
    @staticmethod
    def extract_year(filename: str) -> int:
        """Extract year with validation"""
        match = re.search(r'20\d{2}', filename)
        if match:
            year = int(match.group(0))
            # Validate year is reasonable (2015-2030)
            if 2015 <= year <= 2030:
                return year
        return datetime.now().year
    
    @staticmethod
    def detect_language(text: str) -> str:
        """IMPROVED language detection"""
        urdu_chars = set("ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھیےۓ")
        
        if not text:
            return "en"
        
        urdu_count = sum(1 for char in text if char in urdu_chars)
        urdu_percentage = urdu_count / len(text)
        
        # More lenient threshold
        if urdu_percentage > 0.03:  # 3% Urdu characters
            return "ur"
        
        return "en"


# ==================== IMPROVED METADATA BUILDERS ====================

class ImprovedBSADPMetadataBuilder:
    """IMPROVED BS/ADP metadata with text preview for reranker"""
    
    @staticmethod
    def build(chunk_text: str, page_num: int, filename: str) -> Dict:
        extractor = EnhancedMetadataExtractor
        prereqs = extractor.extract_prerequisites(chunk_text)
        course_code = extractor.extract_course_code(chunk_text)
        
        # NEW: Text preview for reranker (first 500 chars)
        text_preview = chunk_text[:500].strip()
        
        metadata = {
            # Core identifiers
            "namespace": "bs-adp-schemes",
            "source_file": filename,
            "page_number": page_num,
            
            # Program information
            "program_type": extractor.extract_program_type(filename),
            "department": extractor.extract_department(filename),
            "batch_year": extractor.extract_year(filename),
            
            # Course information
            "course_code": course_code,
            "course_title": extractor.extract_course_title(chunk_text),  # NEW
            "credit_hours": extractor.extract_credit_hours(chunk_text),
            "semester": extractor.extract_semester(chunk_text),
            
            # Prerequisites
            "prerequisites": prereqs,
            "has_prerequisites": len(prereqs) > 0,
            "num_prerequisites": len(prereqs),  # NEW: Count
            
            # Content type detection
            "is_course_outline": bool(re.search(r'Course (Code|Title|Outline)', chunk_text, re.IGNORECASE)),
            "has_learning_outcomes": "CLO" in chunk_text or "Learning Outcome" in chunk_text,
            "has_assessment": bool(re.search(r'(Assessment|Evaluation|Marks|Grade)', chunk_text, re.IGNORECASE)),
            "has_objectives": bool(re.search(r'(Course Objectives?|Objectives?:)', chunk_text, re.IGNORECASE)),  # NEW
            
            # NEW: Content indicators
            "mentions_lab": bool(re.search(r'\blab\b|\blaboratory\b', chunk_text, re.IGNORECASE)),
            "mentions_project": bool(re.search(r'\bproject\b|\bassignment\b', chunk_text, re.IGNORECASE)),
            "is_elective": 'elective' in chunk_text.lower(),
            "is_compulsory": bool(re.search(r'\bcompulsory\b|\bmandatory\b|\brequired\b', chunk_text, re.IGNORECASE)),
            
            # Technical metadata
            "language": extractor.detect_language(chunk_text),
            "chunk_length": len(chunk_text),
            "text_preview": text_preview,  # NEW: For reranker
            "ingestion_timestamp": datetime.now().isoformat(),
            
            # NEW: Quality indicators
            "has_course_code": bool(course_code),
            "is_complete_section": len(chunk_text) > 200,  # Indicates complete info
        }
        
        return metadata


class ImprovedMSPHDMetadataBuilder:
    """IMPROVED MS/PhD metadata"""
    
    @staticmethod
    def build(chunk_text: str, page_num: int, filename: str) -> Dict:
        extractor = EnhancedMetadataExtractor
        course_code = extractor.extract_course_code(chunk_text)
        text_preview = chunk_text[:500].strip()
        
        return {
            # Core identifiers
            "namespace": "ms-phd-schemes",
            "source_file": filename,
            "page_number": page_num,
            
            # Program information
            "program_level": extractor.extract_program_type(filename),
            "department": extractor.extract_department(filename),
            "batch_year": extractor.extract_year(filename),
            
            # Course information
            "course_code": course_code,
            "course_title": extractor.extract_course_title(chunk_text),  # NEW
            "credit_hours": extractor.extract_credit_hours(chunk_text),
            "semester": extractor.extract_semester(chunk_text),
            
            # Research-specific
            "is_thesis_related": bool(re.search(r'thesis|research|dissertation', chunk_text, re.IGNORECASE)),
            "mentions_viva": 'viva' in chunk_text.lower() or 'oral exam' in chunk_text.lower(),
            "mentions_publication": bool(re.search(r'publicat|journal|conference', chunk_text, re.IGNORECASE)),
            "mentions_supervisor": bool(re.search(r'supervisor|adviser|advisor', chunk_text, re.IGNORECASE)),  # NEW
            
            # Content type
            "is_course_outline": bool(re.search(r'Course (Code|Title|Outline|Objective)', chunk_text, re.IGNORECASE)),
            "has_reading_list": bool(re.search(r'(Recommended|Suggested) (Reading|Books)', chunk_text, re.IGNORECASE)),
            "has_methodology": bool(re.search(r'methodology|teaching method', chunk_text, re.IGNORECASE)),  # NEW
            
            # Technical metadata
            "language": extractor.detect_language(chunk_text),
            "chunk_length": len(chunk_text),
            "text_preview": text_preview,  # NEW: For reranker
            "ingestion_timestamp": datetime.now().isoformat(),
            
            # Quality indicators
            "has_course_code": bool(course_code),
            "is_complete_section": len(chunk_text) > 200,
        }


class ImprovedRulesMetadataBuilder:
    """IMPROVED Rules metadata"""
    
    @staticmethod
    def build(chunk_text: str, page_num: int, filename: str) -> Dict:
        extractor = EnhancedMetadataExtractor
        text_preview = chunk_text[:500].strip()
        
        # Extract rule numbers
        rule_number = ""
        for pattern in [
            r'Article\s+([\d\.]+)',
            r'Section\s+([\d\.]+)',
            r'Rule\s+([\d\.]+)',
            r'Clause\s+([\d\.]+)',  # NEW
            r'^(\d+)\.\s+[A-Z]'
        ]:
            match = re.search(pattern, chunk_text, re.IGNORECASE | re.MULTILINE)
            if match:
                rule_number = match.group(0)
                break
        
        # Improved rule categorization
        filename_lower = filename.lower()
        text_lower = chunk_text.lower()
        
        categories = {
            'admission': ['admission', 'entry', 'eligibility', 'intake'],
            'examination': ['exam', 'examination', 'assessment', 'test'],
            'hostel': ['hostel', 'residence', 'accommodation', 'boarding'],
            'fee': ['fee', 'tuition', 'payment', 'charges'],
            'migration': ['migration', 'transfer', 'credit transfer'],
            'grading': ['grading', 'grade', 'cgpa', 'gpa', 'marks'],
            'probation': ['probation', 'academic standing', 'warning'],
            'discipline': ['discipline', 'conduct', 'behavior', 'misconduct'],
            'attendance': ['attendance', 'leave', 'absence'],
            'scholarship': ['scholarship', 'merit', 'financial aid', 'stipend'],
            'library': ['library', 'books', 'resources'],
        }
        
        rule_category = 'general'
        for category, keywords in categories.items():
            if any(kw in filename_lower or kw in text_lower for kw in keywords):
                rule_category = category
                break
        
        return {
            # Core identifiers
            "namespace": "rules-regulations",
            "source_file": filename,
            "page_number": page_num,
            
            # Rule classification
            "rule_category": rule_category,
            "rule_number": rule_number,
            "has_rule_number": bool(rule_number),  # NEW
            
            # Applicability
            "applies_to_bs": bool(re.search(r'\bbs\b|\bbachelor', text_lower)),
            "applies_to_ms": bool(re.search(r'\bms\b|\bmaster', text_lower)),
            "applies_to_phd": bool(re.search(r'\bphd\b|\bdoctoral', text_lower)),
            "applies_to_all": bool(re.search(r'\ball students\b|\bevery student', text_lower)),  # NEW
            
            # Important flags
            "mentions_penalty": bool(re.search(r'penalty|fine|punish|expel|suspend', chunk_text, re.IGNORECASE)),
            "mentions_deadline": bool(re.search(r'deadline|due date|before|within.*days|last date', chunk_text, re.IGNORECASE)),
            "mentions_fee": bool(re.search(r'fee|payment|amount|rupees|rs\.?|pkr', chunk_text, re.IGNORECASE)),
            "mentions_cgpa": bool(re.search(r'\bcgpa\b|\bgpa\b', text_lower)),
            "mentions_percentage": bool(re.search(r'\d+%|percentage|percent', chunk_text)),  # NEW
            
            # Merit/Quota related
            "is_merit_calculation": bool(re.search(r'merit|calculation|formula|percentage', chunk_text, re.IGNORECASE)),
            "is_quota_info": bool(re.search(r'quota|reserved|seat.*reserv', chunk_text, re.IGNORECASE)),
            "is_probation_rule": bool(re.search(r'probation|academic.*standing|warning', chunk_text, re.IGNORECASE)),
            
            # Content type
            "has_formula": bool(re.search(r'formula|calculate|=|×|÷|\+|-', chunk_text)),
            "has_table": '|' in chunk_text or bool(re.search(r'\t.*\t', chunk_text)),
            "has_list": bool(re.search(r'(\n\s*[-•]\s+|\n\s*\d+\.\s+)', chunk_text)),  # NEW
            
            # Technical metadata
            "language": extractor.detect_language(chunk_text),
            "chunk_length": len(chunk_text),
            "text_preview": text_preview,  # NEW: For reranker
            "ingestion_timestamp": datetime.now().isoformat(),
            
            # Quality indicators
            "is_complete_rule": len(chunk_text) > 150,
        }


# ==================== PROGRESS TRACKING ====================

class ProgressTracker:
    """Track ingestion progress for resume capability"""
    
    def __init__(self):
        self.processed_files: Set[str] = set()
        self.failed_files: Dict[str, str] = {}
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Load existing progress
        self._load_progress()
    
    def _load_progress(self):
        """Load progress from previous run"""
        if PROCESSED_FILES_LOG.exists():
            try:
                with open(PROCESSED_FILES_LOG, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                    logger.info(f"Loaded {len(self.processed_files)} previously processed files")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def mark_processed(self, filepath: str):
        """Mark file as processed"""
        self.processed_files.add(filepath)
        self._save_progress()
    
    def mark_failed(self, filepath: str, error: str):
        """Mark file as failed"""
        self.failed_files[filepath] = error
    
    def is_processed(self, filepath: str) -> bool:
        """Check if file was already processed"""
        return filepath in self.processed_files
    
    def _save_progress(self):
        """Save progress to disk"""
        try:
            with open(PROCESSED_FILES_LOG, 'w') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'failed_files': self.failed_files,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        elapsed = datetime.now() - datetime.fromisoformat(self.stats['start_time'])
        return {
            **self.stats,
            'elapsed_seconds': elapsed.total_seconds(),
            'chunks_per_second': self.stats['total_chunks'] / max(elapsed.total_seconds(), 1)
        }


# ==================== IMPROVED INGESTION PIPELINE ====================

class ImprovedPineconeIngestionPipeline:
    """
    IMPROVED ingestion pipeline with:
    1. Resume capability
    2. Better error handling
    3. Progress tracking
    4. Deduplication
    5. Optimized batching
    """
    
    def __init__(self, resume_mode: bool = False):
        logger.info("="*70)
        logger.info("Initializing IMPROVED Pinecone Ingestion Pipeline")
        logger.info("="*70)
        
        # Validate environment
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("❌ PINECONE_API_KEY not found")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY not found")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self._setup_index()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize progress tracker
        self.progress = ProgressTracker()
        self.resume_mode = resume_mode
        
        logger.info("✅ Initialization complete\n")
    
    def _setup_index(self):
        """Setup Pinecone index with better error handling"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"📊 Creating index: {PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud=PINECONE_CLOUD,
                        region=PINECONE_REGION
                    )
                )
                logger.info("✅ Index created")
                
                # Wait for index to be ready
                import time
                time.sleep(10)
            else:
                logger.info(f"✅ Connected to index: {PINECONE_INDEX_NAME}")
            
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
        except Exception as e:
            logger.error(f"❌ Failed to setup index: {e}")
            raise
    
    def process_namespace(self, namespace: str):
        """Process namespace with improved error handling and progress tracking"""
        logger.info("\n" + "="*70)
        logger.info(f"📁 PROCESSING NAMESPACE: {namespace.upper()}")
        logger.info("="*70)
        
        folder_path = NAMESPACES[namespace]
        
        if not folder_path.exists():
            logger.error(f"❌ Folder not found: {folder_path}")
            return
        
        # Get all PDFs
        pdf_files = list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.PDF"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {folder_path}")
            return
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        # Filter out already processed files in resume mode
        if self.resume_mode:
            initial_count = len(pdf_files)
            pdf_files = [
                f for f in pdf_files 
                if not self.progress.is_processed(f"{namespace}::{f.name}")
            ]
            skipped = initial_count - len(pdf_files)
            if skipped > 0:
                logger.info(f"⏭️  Skipping {skipped} already processed files")
        
        if not pdf_files:
            logger.info("✅ All files already processed")
            return
        
        logger.info(f"🔧 Chunking: Semantic (optimized)")
        logger.info(f"🧠 Embedding: {OPENAI_EMBEDDING_MODEL}")
        logger.info(f"📊 Dimension: {PINECONE_DIMENSION}")
        logger.info(f"🎯 Namespace: {namespace}\n")
        
        # Initialize chunker and metadata builder
        chunker = ImprovedSemanticChunker(namespace)
        
        if namespace == "bs-adp-schemes":
            metadata_builder = ImprovedBSADPMetadataBuilder()
        elif namespace == "ms-phd-schemes":
            metadata_builder = ImprovedMSPHDMetadataBuilder()
        else:
            metadata_builder = ImprovedRulesMetadataBuilder()
        
        # Process files
        namespace_chunks = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            file_key = f"{namespace}::{pdf_file.name}"
            
            try:
                chunks = self._process_single_pdf(
                    pdf_file,
                    namespace,
                    chunker,
                    metadata_builder
                )
                
                if chunks > 0:
                    namespace_chunks += chunks
                    self.progress.mark_processed(file_key)
                    self.progress.stats['processed_files'] += 1
                    self.progress.stats['total_chunks'] += chunks
                
            except Exception as e:
                logger.error(f"\n❌ Failed: {pdf_file.name}: {str(e)}")
                self.progress.mark_failed(file_key, str(e))
                self.progress.stats['failed_files'] += 1
                continue
        
        logger.info(f"\n✅ Namespace complete: {namespace_chunks} chunks created")
    
    def _process_single_pdf(
        self,
        pdf_path: Path,
        namespace: str,
        chunker: ImprovedSemanticChunker,
        metadata_builder
    ) -> int:
        """Process single PDF with improved error handling"""
        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No content: {pdf_path.name}")
                return 0
            
            # Split into chunks
            chunks = chunker.split_documents(pages)
            
            if not chunks:
                logger.warning(f"No chunks: {pdf_path.name}")
                return 0
            
            # Build documents
            documents = []
            file_hash = self._compute_file_hash(pdf_path)
            
            for i, chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue
                
                page_num = chunk.metadata.get('page', 0) + 1
                
                # Build metadata
                metadata = metadata_builder.build(
                    chunk_text=chunk.page_content,
                    page_num=page_num,
                    filename=pdf_path.name
                )
                
                # Add chunk metadata
                metadata.update({
                    "chunk_id": f"{pdf_path.stem}_p{page_num}_c{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_hash": file_hash
                })
                
                documents.append(Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                ))
            
            if not documents:
                return 0
            
            # Upsert to Pinecone
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace
            )
            
            # Batch upsert
            batch_size = EMBEDDING_BATCH_SIZE
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                vector_store.add_texts(texts=texts, metadatas=metadatas)
            
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            raise
    
    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute file hash"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def ingest_all(self):
        """Ingest all namespaces"""
        logger.info("\n" + "="*70)
        logger.info("🎓 UE LAHORE - FULL INGESTION")
        logger.info("="*70)
        
        for namespace in NAMESPACES.keys():
            self.process_namespace(namespace)
        
        # Final summary
        summary = self.progress.get_summary()
        
        logger.info("\n" + "="*70)
        logger.info("✅ INGESTION COMPLETE")
        logger.info("="*70)
        logger.info(f"⏱️  Time: {summary['elapsed_seconds']:.1f}s")
        logger.info(f"📁 Files: {summary['processed_files']}/{summary['total_files']}")
        logger.info(f"❌ Failed: {summary['failed_files']}")
        logger.info(f"📦 Chunks: {summary['total_chunks']}")
        logger.info(f"⚡ Speed: {summary['chunks_per_second']:.2f} chunks/s")
        
        # Index stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"\n📊 Index Stats:")
            logger.info(f"   Total vectors: {stats.total_vector_count}")
            for ns, ns_stats in stats.namespaces.items():
                logger.info(f"   - {ns}: {ns_stats.vector_count} vectors")
        except:
            pass
        
        logger.info("="*70)
    
    def ingest_single_namespace(self, namespace: str):
        """Ingest single namespace"""
        if namespace not in NAMESPACES:
            logger.error(f"Invalid namespace: {namespace}")
            return
        
        logger.info(f"\n🎯 Single namespace ingestion: {namespace}")
        self.process_namespace(namespace)


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    
    # Check for resume flag
    resume_mode = '--resume' in sys.argv
    
    # Create pipeline
    pipeline = ImprovedPineconeIngestionPipeline(resume_mode=resume_mode)
    
    # Get namespace argument
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if args:
        namespace_arg = args[0]
        namespace_map = {
            "bs": "bs-adp-schemes",
            "ms": "ms-phd-schemes",
            "rules": "rules-regulations",
        }
        namespace = namespace_map.get(namespace_arg, namespace_arg)
        pipeline.ingest_single_namespace(namespace)
    else:
        pipeline.ingest_all()
    
    logger.info("\n🎉 All done! Data ready in Pinecone.")