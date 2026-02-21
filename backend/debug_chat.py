import sys
import logging
import traceback

# Configure logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("--- STARTING DEBUG SCRIPT ---")

try:
    print("1. Importing rag_pipeline...")
    from rag_pipeline import get_pipeline
    
    print("2. Initializing pipeline...")
    pipeline = get_pipeline()
    print("   Pipeline initialized.")

    print("3. Running query (standard mode)...")
    # mimicking the request that caused 500
    try:
        result = pipeline.query(
            user_query="test query",
            namespace="bs-adp",
            enhance_query=True,
            top_k_retrieve=6,
            session_id="debug-session",
            enable_smart=False
        )
        print("4. Query SUCCESS!")
        print("   Answer:", result.get("answer")[:50], "...")
    except Exception as e:
        print("\n!!! QUERY FAILED !!!")
        traceback.print_exc()

    print("5. Running query (enable_smart=True)...")
    try:
        result = pipeline.query(
            user_query="test query",
            namespace="bs-adp",
            enhance_query=True,
            top_k_retrieve=6,
            session_id="debug-session",
            enable_smart=True
        )
        print("6. Smart Query SUCCESS!")
        print("   Answer:", result.get("answer")[:50], "...")
    except Exception as e:
        print("\n!!! SMART QUERY FAILED !!!")
        traceback.print_exc()

except ImportError as e:
    print("\n!!! IMPORT FAILED !!!")
    traceback.print_exc()
except Exception as e:
    print("\n!!! SETUP FAILED !!!")
    traceback.print_exc()

print("--- END DEBUG SCRIPT ---")
