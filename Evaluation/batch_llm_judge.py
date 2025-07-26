import os
import glob
import subprocess
import sys
from pathlib import Path

def get_csv_files(raw_csv_dir):
    """Get all CSV files from the raw_csv directory."""
    csv_pattern = os.path.join(raw_csv_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    return [os.path.basename(f) for f in csv_files]

def run_llm_judge(csv_filename):
    """Run the LLM judge script for a single CSV file."""
    print(f"\n{'='*60}")
    print(f"Processing: {csv_filename}")
    print(f"{'='*60}")
    
    try:
        # Get the absolute path to the llm_as_judge.py script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        llm_judge_script = os.path.join(script_dir, "llm_as_judge.py")
        
        # Run the llm_as_judge.py script
        result = subprocess.run([
            sys.executable, 
            llm_judge_script, 
            "--input_csv", 
            csv_filename
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ Successfully processed: {csv_filename}")
            print(result.stdout)
        else:
            print(f"❌ Error processing: {csv_filename}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception while processing {csv_filename}: {e}")
        return False
    
    return True

def main():
    """Main function to process all CSV files."""
    # Get the raw_csv directory path
    raw_csv_dir = "data/raw_csv"
    
    # Check if directory exists
    if not os.path.exists(raw_csv_dir):
        print(f"❌ Directory '{raw_csv_dir}' does not exist!")
        print("Please create the directory and place your CSV files there.")
        return
    
    # Get all CSV files
    csv_files = get_csv_files(raw_csv_dir)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{raw_csv_dir}'!")
        print("Please place your CSV files in the data/raw_csv/ directory.")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files to process:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    # Ask for confirmation
    response = input(f"\nDo you want to process all {len(csv_files)} files? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user.")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n🔄 Processing file {i}/{len(csv_files)}: {csv_file}")
        
        if run_llm_judge(csv_file):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed to process: {failed} files")
    print(f"📁 Total files: {len(csv_files)}")
    
    if successful > 0:
        print(f"\n📂 Results saved in: data/result_csv/")
        print("Files will have '_llm_judge_results.csv' suffix.")

if __name__ == "__main__":
    main() 