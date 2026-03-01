import traceback,re
from transformers import AutoModelForCausalLM
from pathlib import Path
import transformers.models.mimi.modeling_mimi as mm


def check_openelm_error():
    try:
        model=AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M",trust_remote_code=True)
    except Exception as e:
        m = re.search(r'File "([^"]+)".*in _compute_sin_cos_embeddings', traceback.format_exc())
        if m:
            p = Path(m.group(1))
            otxt = 'self._compute_sin_cos_embeddings(max_seq_length)'
            ntxt = '# self._compute_sin_cos_embeddings (max_seq_length)'
            p.write_text(p.read_text().replace(otxt, ntxt))
        model=AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M",trust_remote_code=True)
    if model:
        print("Success.")

def check_mimi_error():
    p = Path(mm.__file__)
    text = p.read_text()
    target = '''"""Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """'''
    insert_line = '\n        paddings = tuple(int(p) for p in paddings)  # !!! patch'

    if insert_line.strip() in text:
        return
    if target not in text:
        print("Docstring not found. Exact match failed.")
        return

    new_text = text.replace(target, target + insert_line)
    p.write_text(new_text)
    print("Patch applied.")

check_openelm_error()
check_mimi_error()