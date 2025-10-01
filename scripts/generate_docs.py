#!/usr/bin/env python
"""
Automatic documentation generator for KMR package.

This script generates markdown documentation from docstrings of all layers and models.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_docstring_info(file_path: Path) -> Dict:
    """Extract class and method information from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {}
    
    classes = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node) or "No documentation available.",
                'methods': []
            }
            
            # Extract method docstrings
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_doc = ast.get_docstring(item) or "No documentation available."
                    class_info['methods'].append({
                        'name': item.name,
                        'docstring': method_doc
                    })
            
            classes[node.name] = class_info
    
    return classes


def generate_layer_docs() -> str:
    """Generate documentation for all layers."""
    layers_dir = Path("kmr/layers")
    docs = ["# KMR Layers Documentation\n"]
    
    layer_files = sorted([f for f in layers_dir.glob("*.py") if f.name != "__init__.py" and f.name != "_base_layer.py"])
    
    for layer_file in layer_files:
        layer_name = layer_file.stem
        classes = extract_docstring_info(layer_file)
        
        if not classes:
            continue
            
        docs.append(f"## {layer_name}\n")
        
        for class_name, class_info in classes.items():
            docs.append(f"### {class_name}\n")
            docs.append(f"{class_info['docstring']}\n")
            
            if class_info['methods']:
                docs.append("#### Methods\n")
                for method in class_info['methods']:
                    if not method['name'].startswith('_'):
                        docs.append(f"**{method['name']}**\n")
                        docs.append(f"{method['docstring']}\n")
        
        docs.append("---\n")
    
    return "\n".join(docs)


def generate_model_docs() -> str:
    """Generate documentation for all models."""
    models_dir = Path("kmr/models")
    docs = ["# KMR Models Documentation\n"]
    
    model_files = sorted([f for f in models_dir.glob("*.py") if f.name != "__init__.py" and f.name != "_base.py"])
    
    for model_file in model_files:
        model_name = model_file.stem
        classes = extract_docstring_info(model_file)
        
        if not classes:
            continue
            
        docs.append(f"## {model_name}\n")
        
        for class_name, class_info in classes.items():
            docs.append(f"### {class_name}\n")
            docs.append(f"{class_info['docstring']}\n")
            
            if class_info['methods']:
                docs.append("#### Methods\n")
                for method in class_info['methods']:
                    if not method['name'].startswith('_'):
                        docs.append(f"**{method['name']}**\n")
                        docs.append(f"{method['docstring']}\n")
        
        docs.append("---\n")
    
    return "\n".join(docs)


def generate_api_overview() -> str:
    """Generate API overview with all available components."""
    docs = ["# KMR API Overview\n"]
    
    # Layers
    layers_dir = Path("kmr/layers")
    layer_files = sorted([f for f in layers_dir.glob("*.py") if f.name != "__init__.py" and f.name != "_base_layer.py"])
    
    docs.append("## Available Layers\n")
    for layer_file in layer_files:
        layer_name = layer_file.stem
        classes = extract_docstring_info(layer_file)
        
        for class_name, class_info in classes.items():
            # Extract first line of docstring as summary
            summary = class_info['docstring'].split('\n')[0] if class_info['docstring'] else "No description available."
            docs.append(f"- **{class_name}**: {summary}\n")
    
    # Models
    models_dir = Path("kmr/models")
    model_files = sorted([f for f in models_dir.glob("*.py") if f.name != "__init__.py" and f.name != "_base.py"])
    
    docs.append("\n## Available Models\n")
    for model_file in model_files:
        model_name = model_file.stem
        classes = extract_docstring_info(model_file)
        
        for class_name, class_info in classes.items():
            summary = class_info['docstring'].split('\n')[0] if class_info['docstring'] else "No description available."
            docs.append(f"- **{class_name}**: {summary}\n")
    
    return "\n".join(docs)


def main():
    """Generate all documentation files."""
    print("Generating KMR documentation...")
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate API overview
    api_docs = generate_api_overview()
    with open(docs_dir / "api_overview.md", "w", encoding="utf-8") as f:
        f.write(api_docs)
    print("✓ Generated API overview")
    
    # Generate layer documentation
    layer_docs = generate_layer_docs()
    with open(docs_dir / "layers.md", "w", encoding="utf-8") as f:
        f.write(layer_docs)
    print("✓ Generated layers documentation")
    
    # Generate model documentation
    model_docs = generate_model_docs()
    with open(docs_dir / "models.md", "w", encoding="utf-8") as f:
        f.write(model_docs)
    print("✓ Generated models documentation")
    
    print("Documentation generation complete!")


if __name__ == "__main__":
    main()
