import os
from simple_tokenizer import SimpleTokenizer
from code_tokenizer import CodeTokenizer

class TokenizerCache2:
    def __init__(self):
        self.st = SimpleTokenizer()
        self.ct = CodeTokenizer()
        self._cache = {}
    
    def get_tokens(self, source_code, tokenizer_type):
        key = (hash(source_code), tokenizer_type)
        if key not in self._cache:
            if tokenizer_type == "simple":
                self._cache[key] = self.st.tokenize(source_code).data
            elif tokenizer_type == "code":
                self._cache[key] = self.ct.tokenize(source_code).data
        return self._cache[key]

def run2(source_code, cache=None):
    # Use a caching mechanism to avoid repeated initialization
    cache = cache or TokenizerCache2()
    
    # The first stage word segmentation optimization
    st_tokens = cache.get_tokens(source_code, "simple")
    code = ' '.join(st_tokens)  # Automatic processing space
    
    # The second stage word segmentation optimization
    ct_tokens = cache.get_tokens(code, "code")
    return ' '.join(ct_tokens)  # Automatic processing space

if __name__ == '__main__':
    pass
