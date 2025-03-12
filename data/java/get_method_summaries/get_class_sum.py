import re
import jsonlines
import torch, gc
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-multi-sum").to(device)
model.eval()
buffer = []


def remove_space(text):
    pattern = r"(.*)[ ](.)"
    result = re.sub(pattern, r"\1\2", text)
    return result

# Extraction method
def extract_methods(java_code):
    # Match method code snippets using regular expressions
    pattern = r"((?:public|private|protected|static|\s)*[\w<>]+\s+[\w]+\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{[\s\S]*?\})"

    methods = re.findall(pattern, java_code, re.MULTILINE)

    return methods


def extract_func(item):
    code = ""
    code += item
    methods = extract_methods(code)

    return methods

def flush_buffer():
    global buffer
    if buffer:
        with open('data/train/method_comment.txt', 'a', encoding='utf-8') as f:
            f.writelines(buffer)
        buffer = []

# Generate method annotations using the CodeT5 model
def get_comment(methods):
    global buffer
    if len(methods)>0:
        if len(methods) < 100:
            inputs = tokenizer(methods, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            generated_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=20)
            comments = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else :
            comments = []
            for method in methods:
                inputs = tokenizer(method, padding=True, truncation=True, max_length=512,return_tensors="pt").to(device)
                generated_ids = model.generate(inputs.input_ids, max_length=20)
                comment = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                comment = re.sub(r'[\r\t\n]', '', comment)
                comment = remove_space(comment)
                comments.append(comment)


        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        str=""
        for comment in comments:
            comment = re.sub(r'[\r\t\n]', '', comment)
            comment = remove_space(comment)
            if (comment[-1] != '.'):
                comment += '.'
            str += comment + ' '
        buffer.append(str + '\n')
    else:
        buffer.append('No methods!\n')
    if len(buffer) >= 500:
        with open('data/train/method_comment.txt', 'a', encoding='utf-8') as f:
            f.writelines(buffer)
        buffer = []


def process_java_classes(input_file, start_line, end_line):
    """Manually specify a range of processing line numbers (including start and end lines)"""
    global buffer
    with open(input_file, 'r', encoding='utf-8') as f:
        # Create a progress bar (total number of rows to process)
        pbar = tqdm(total=end_line - start_line + 1, desc="Processing lines")

        for line_num, line in enumerate(f, 1):  # Line numbers count from 1
            # Only rows within the specified range are processed
            if start_line <= line_num <= end_line:
                methods = extract_methods(line.strip())
                get_comment(methods)
                pbar.update(1)

            # Stop processing when the end line is exceeded
            if line_num > end_line:
                break

        # Write remaining buffer
        if buffer:
            with open('data/train/method_comment.txt', 'a', encoding='utf-8') as f:
                f.writelines(buffer)
            buffer = []

# Execute and output the result
# input_file should be originalcode.txt in the data folder
input_file = "data/train/originalcode.txt"
process_java_classes(
    input_file=input_file,
    start_line=1,    # Start line number (inclusive)
    end_line=252006   # End line number (inclusive)
)