"""Main equation generation logic."""

import json
import os
import time
from config import DATASET_PATH, get_llm_config, get_output_path
from llm_client import create_client
from utils import build_combined_context, construct_final_prompt


class MathGenerator:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.config, self.api_key = get_llm_config(llm_name)
        self.output_path = get_output_path(llm_name)
        self.generate_fn = create_client(self.config, self.api_key)
        
        # Load data
        self.equations = self.load_dataset()
        self.results = self.load_results()
        
        print(f"🤖 Using {llm_name} ({self.config.model})")
        self.print_stats()
    
    def load_dataset(self):
        """Load equations from dataset."""
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        equations = []
        for paper in dataset:
            for eq in paper['equations']:
                equations.append({
                    'paper_id': paper['id'],
                    'equation_id': eq['equation id'],
                    'context': eq['context'],
                    'description': eq.get('description', ''),
                    'EQ_latex': eq.get('EQ_latex', [])
                })
        return equations
    
    def load_results(self):
        """Load existing results."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_results(self):
        """Save results to file."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def print_stats(self):
        """Print progress statistics."""
        total = len(self.equations)
        done = sum(len(paper) for paper in self.results.values())
        pending = total - done
        progress = (done / total * 100) if total > 0 else 0
        
        print(f"📊 Progress: {done}/{total} ({progress:.1f}%) | Pending: {pending}")
    
    def get_pending_equations(self, paper_ids=None):
        """Get equations that need to be processed."""
        pending = []
        for eq in self.equations:
            paper_id = eq['paper_id']
            equation_id = eq['equation_id']
            
            # Filter by paper_ids if specified
            if paper_ids and paper_id not in paper_ids:
                continue
            
            # Skip if already done
            if paper_id in self.results and equation_id in self.results[paper_id]:
                continue
            
            pending.append(eq)
        return pending
    
    def generate_all(self, fresh=False, paper_ids=None):
        """Generate all pending equations."""
        if fresh:
            self.results = {}
            print("🆕 Starting fresh generation")
        
        pending = self.get_pending_equations(paper_ids)
        if not pending:
            print("✅ All equations already generated!")
            return
        
        print(f"🔄 Generating {len(pending)} equations...")
        
        blocks = build_combined_context(pending)
        generated = 0
        
        for paper_id, eq_id, context in blocks:
            print(f"Processing {paper_id}-{eq_id}...", end=" ")
            
            try:
                prompt = construct_final_prompt(context)
                latex = self.generate_fn(prompt)
                
                # Store result
                if paper_id not in self.results:
                    self.results[paper_id] = {}
                self.results[paper_id][eq_id] = latex.strip()
                
                generated += 1
                print(f"✅ {latex[:50]}{'...' if len(latex) > 50 else ''}")
                
                # Save every 5 equations
                if generated % 5 == 0:
                    self.save_results()
                    print(f"💾 Saved progress ({generated} done)")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        self.save_results()
        print(f"✅ Generated {generated} equations!")
        self.print_stats()
    
    def show_status(self):
        """Show current status and sample results."""
        self.print_stats()
        
        if self.results:
            print("\n📋 Sample Results:")
            count = 0
            for paper_id, equations in self.results.items():
                for eq_id, latex in equations.items():
                    if count >= 3:
                        break
                    print(f"  {paper_id}-{eq_id}: {latex[:100]}{'...' if len(latex) > 100 else ''}")
                    count += 1
                if count >= 3:
                    break