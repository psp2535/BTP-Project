import os
import re

def verify_latex(file_path):
    print(f"Verifying LaTeX file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Check balanced environments
    begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', content)
    ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', content)
    
    print(f"Total \\begin count: {len(begins)}, Total \\end count: {len(ends)}")
    if len(begins) != len(ends):
        print("[ERROR] Mismatch in environment counts!")
    
    # Check stack
    stack = []
    lines = content.split('\n')
    for idx, line in enumerate(lines):
        for m in re.finditer(r'\\begin\{([a-zA-Z0-9*]+)\}', line):
            stack.append((m.group(1), idx + 1))
        for m in re.finditer(r'\\end\{([a-zA-Z0-9*]+)\}', line):
            if not stack:
                print(f"[ERROR] Line {idx+1}: Unexpected \\end{{{m.group(1)}}}")
            else:
                last, last_line = stack.pop()
                if last != m.group(1):
                    print(f"[ERROR] Line {idx+1}: Mismatched \\end{{{m.group(1)}}}, expected \\end{{{last}}} from line {last_line}")
                    
    if stack:
        for env, l_num in stack:
            print(f"[ERROR] Unclosed \\begin{{{env}}} from line {l_num}")
    else:
        print("[PASSED] All LaTeX environments are perfectly balanced!")

    # 2. Check Image Files
    img_refs = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', content)
    print(f"\nChecking {len(img_refs)} figure inclusions:")
    for img in img_refs:
        exists = os.path.exists(img)
        status = "[EXISTS]" if exists else "[MISSING]"
        print(f"  {status} {img}")
        if not exists:
            print(f"    [WARNING] Image file {img} not found on disk!")

    # 3. Check Labels and References
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    print(f"\nTotal \\label: {len(labels)}, Total \\ref: {len(refs)}")
    missing_labels = refs - labels
    if missing_labels:
        print(f"[WARNING] References without matching labels: {missing_labels}")
    else:
        print("[PASSED] All \\ref tags match existing \\label definitions!")

    # 4. Check Citations and Bibliography
    bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', content))
    citations = set(re.findall(r'\\cite\{([^}]+)\}', content))
    # Note citations can be comma separated
    expanded_cites = set()
    for c in citations:
        for sub in c.split(','):
            expanded_cites.add(sub.strip())
            
    print(f"\nTotal \\bibitem entries: {len(bibitems)}, Total \\cite keys: {len(expanded_cites)}")
    missing_cites = expanded_cites - bibitems
    if missing_cites:
        print(f"[WARNING] Citations without matching bibitems: {missing_cites}")
    else:
        print("[PASSED] All \\cite keys match existing \\bibitem entries!")

if __name__ == '__main__':
    verify_latex('IM-VRM_Paper.tex')
