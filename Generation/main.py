"""Command-line interface for math generation."""

import argparse
import sys
from config import LLMS
from generator import MathGenerator

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX equations using different LLMs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--llm', type=str, help='LLM to use for generation (e.g., "gpt-4o")')
    parser.add_argument('--list', action='store_true', help='List available LLMs')
    parser.add_argument('--fresh', action='store_true', help='Start fresh, ignoring any previous results')
    parser.add_argument('--papers', nargs='+', help='Generate only for specific paper IDs (e.g., --papers "2024.acl-short.1")')
    parser.add_argument('--status', action='store_true', help='Show the current progress and status')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available LLMs:")
        print("-" * 60)
        for name, config in LLMS.items():
            print(f"  - {name:15} (Provider: {config.provider})")
        print("-" * 60)
        return

    if not args.llm:
        parser.print_help()
        sys.exit(1)

    if args.llm not in LLMS:
        print(f"❌ Error: Unknown LLM '{args.llm}'")
        print("Use --list to see available options.")
        sys.exit(1)
    
    generator = None
    exit_code = 0
    try:
        print(f"🔧 Initializing generator for '{args.llm}'...")
        generator = MathGenerator(args.llm)
        
        if args.status:
            generator.show_status()
        else:
            print("🚀 Starting generation...")
            generator.generate_all(fresh=args.fresh, paper_ids=args.papers)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user.")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        exit_code = 1
    finally:
        if generator and not args.status:
            print("\n💾 Attempting to save final results...")
            generator.save_results()
            print(f"✅ Results saved to: {generator.output_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()