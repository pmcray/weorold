import json

def strip_notebook(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Clear outputs
        for cell in nb.get('cells', []):
            if 'outputs' in cell:
                cell['outputs'] = []
            if 'execution_count' in cell:
                cell['execution_count'] = None
                
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
        print(f"Stripped notebook: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    strip_notebook('weorold.ipynb')
