import logging
import re
import subprocess
import tempfile
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def calculate_similarity(a: str, b: str) -> float:
    """Calculate basic sequence similarity ratio."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-style text."""
    blocks = re.findall(r'```(?:[a-zA-Z0-9\#\+]+)?\n(.*?)\n```', text, re.DOTALL)
    return blocks if blocks else [text]

def diff_assembler(gen_cpp: str, original_asm: str) -> float:
    """
    Compiles generated C++ to assembly and compares it with original assembly.
    Returns a similarity score between 0 and 1.
    """
    try:
        # Use temp files for compilation
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as cpp_file:
            cpp_file.write(gen_cpp.encode())
            cpp_filename = cpp_file.name
        
        asm_filename = cpp_filename.replace(".cpp", ".s")
        # Try to compile to assembly using clang or g++
        # Adding -O2 to get somewhat optimized assembly
        result = subprocess.run(
            ["clang", "-S", "-O2", cpp_filename, "-o", asm_filename],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.warning("Clang compilation failed during assembler diffing: %s", result.stderr)
            return 0.0
            
        with open(asm_filename, "r") as f:
            gen_asm = f.read()
            
        # Basic cleaning of assembly (remove comments, labels if needed)
        gen_asm_clean = "\n".join([line.split(';')[0].strip() for line in gen_asm.splitlines() if line.strip()])
        orig_asm_clean = "\n".join([line.split(';')[0].strip() for line in original_asm.splitlines() if line.strip()])
        
        return calculate_similarity(gen_asm_clean, orig_asm_clean)
        
    except Exception as e:
        logger.error("Error in diff_assembler: %s", str(e))
        return 0.0

def diff_cfg(gen_cpp: str, original_binary_path: str, func_name: str) -> float:
    """
    Compares the Control Flow Graph (CFG) of generated C++ vs original binary.
    Requires 'angr' and a compiler.
    """
    try:
        import angr
        import networkx as nx
        
        # 1. Compile generated C++ to a temporary binary
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as cpp_file:
            cpp_file.write(gen_cpp.encode())
            cpp_filename = cpp_file.name
        
        bin_filename = cpp_filename.replace(".cpp", ".out")
        subprocess.run(["clang", "-O2", cpp_filename, "-o", bin_filename], check=True)
        
        # 2. Extract CFG from generated binary
        proj_gen = angr.Project(bin_filename, auto_load_libs=False)
        cfg_gen = proj_gen.analyses.CFGFast()
        # Find function in generated binary
        func_gen = None
        for addr, func in cfg_gen.kb.functions.items():
            if func_name in func.name:
                func_gen = func
                break
        
        if not func_gen:
            logger.warning("Function %s not found in generated binary CFG", func_name)
            return 0.0

        # 3. Extract CFG from original binary
        proj_orig = angr.Project(original_binary_path, auto_load_libs=False)
        cfg_orig = proj_orig.analyses.CFGFast()
        func_orig = None
        for addr, func in cfg_orig.kb.functions.items():
            if func_name in func.name:
                func_orig = func
                break
                
        if not func_orig:
            logger.warning("Function %s not found in original binary CFG", func_name)
            return 0.0
            
        # 4. Compare graph structures (nodes count, edges count, cyclomatic complexity)
        # Using a very basic metric for now: (min(N1, N2) / max(N1, N2) + min(E1, E2) / max(E1, E2)) / 2
        nodes_gen = len(func_gen.transition_graph.nodes())
        nodes_orig = len(func_orig.transition_graph.nodes())
        edges_gen = len(func_gen.transition_graph.edges())
        edges_orig = len(func_orig.transition_graph.edges())
        
        node_sim = min(nodes_gen, nodes_orig) / max(nodes_gen, nodes_orig) if max(nodes_gen, nodes_orig) > 0 else 1.0
        edge_sim = min(edges_gen, edges_orig) / max(edges_gen, edges_orig) if max(edges_gen, edges_orig) > 0 else 1.0
        
        return (node_sim + edge_sim) / 2.0
        
    except ImportError:
        logger.debug("angr or networkx not installed, skipping CFG diff.")
        return 0.0
    except Exception as e:
        logger.error("Error in diff_cfg: %s", str(e))
        return 0.0

def run_similarity_analysis(generated_text: str, source_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Runs multi-level similarity analysis.
    
    Args:
        generated_text: The output from the LLM.
        source_text: The input query/source code.
        context: Optional dict containing 'original_asm', 'binary_path', 'func_name', etc.
    """
    gen_blocks = extract_code_blocks(generated_text)
    src_blocks = extract_code_blocks(source_text)
    
    gen_code = gen_blocks[0] if gen_blocks else generated_text
    src_code = src_blocks[0] if src_blocks else source_text
    
    # Textual Similarity
    text_sim = calculate_similarity(gen_code, src_code)
    
    metrics = {
        "text_similarity": text_sim,
        "assembler_similarity": 0.0,
        "cfg_similarity": 0.0
    }
    
    if context:
        original_asm = context.get("original_asm")
        if original_asm:
            metrics["assembler_similarity"] = diff_assembler(gen_code, original_asm)
            
        binary_path = context.get("binary_path")
        func_name = context.get("func_name")
        if binary_path and func_name:
            metrics["cfg_similarity"] = diff_cfg(gen_code, binary_path, func_name)
            
    # Calculate overall confidence
    weights = {"text_similarity": 0.3, "assembler_similarity": 0.4, "cfg_similarity": 0.3}
    overall = sum(metrics[k] * weights[k] for k in weights)
    metrics["overall_score"] = overall
    
    return metrics
