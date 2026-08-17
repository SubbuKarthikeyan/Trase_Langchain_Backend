import os
import tempfile
import time
from app.rag.build_index import check_and_update_index
from app.rag.vectordb import VectorDB


def test_incremental_ingestion():
    print("\n--- Running Incremental Ingestion Test ---")
    
    db = VectorDB()
    
    # 1. Run index update on the default data/raw directory
    print("\n[Step 1] Initial check_and_update_index call...")
    check_and_update_index("data/raw")
    
    sources_after_step1 = db.get_unique_sources()
    print(f"Sources in MongoDB after Step 1: {list(sources_after_step1.keys())}")
    assert len(sources_after_step1) > 0, "No sources were ingested!"
    
    # 2. Run index update again without changing any file
    print("\n[Step 2] Second check_and_update_index call (should skip all)...")
    check_and_update_index("data/raw")
    
    sources_after_step2 = db.get_unique_sources()
    assert sources_after_step1 == sources_after_step2, "Sources changed unexpectedly!"
    
    # 3. Create a temporary test file in data/raw
    test_file_path = os.path.join("data/raw", "temp_test_doc.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("This is a temporary test document for incremental vector db ingestion.\nLine 2 info.")
        
    try:
        print("\n[Step 3] Running check_and_update_index with new temp file...")
        check_and_update_index("data/raw")
        
        sources_after_step3 = db.get_unique_sources()
        print(f"Sources in MongoDB after Step 3: {list(sources_after_step3.keys())}")
        assert "temp_test_doc.txt" in sources_after_step3, "temp_test_doc.txt was not ingested!"
        
    finally:
        # Clean up temporary test file from disk
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            
    # 4. Run index update after removing temp file from disk (should prune it from MongoDB)
    print("\n[Step 4] Running check_and_update_index after deleting temp file from disk...")
    check_and_update_index("data/raw")
    
    sources_after_step4 = db.get_unique_sources()
    print(f"Sources in MongoDB after Step 4: {list(sources_after_step4.keys())}")
    assert "temp_test_doc.txt" not in sources_after_step4, "temp_test_doc.txt was not pruned from MongoDB!"

    
    print("\nSUCCESS: All incremental ingestion tests passed!")


if __name__ == "__main__":
    test_incremental_ingestion()
