import traceback,re
from transformers import AutoModelForCausalLM
from pathlib import Path

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

check_openelm_error()