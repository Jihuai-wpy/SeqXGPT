import transformers
import re
import numpy as np

from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin


def tokenize_and_mask(text, span_length=2, pct=0.1, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")])
        for text in texts
    ]


def replace_masks(texts, tokenizer, model, mask_top_p=0.95, DEVICE='cuda:0'):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = model.generate(**tokens,
                             max_length=512,
                             do_sample=True,
                             top_p=mask_top_p,
                             num_return_sequences=1,
                             eos_token_id=stop_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills,
              n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts(text, tokenizer, model, ptb_nums, span_length=2, pct=0.3):
    texts = [text for i in range(0, ptb_nums)]
    masked_texts = [tokenize_and_mask(x, span_length, pct) for x in texts]
    raw_fills = replace_masks(masked_texts, tokenizer, model)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    perturbed_texts = [text for text in perturbed_texts if text != '']
    return perturbed_texts


class T5(MsgpackMixin, Worker):

    def __init__(self):
        """Init the model for inference."""
        self.device = 'cuda'

        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            't5-3b')
        self.base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            't5-3b')
        self.base_model.to(self.device)

    def forward(self, data):
        """Override the forward process."""
        """Use T5 to generate multiple rebuild texts """
        data = data['text']
        generated_texts = []
        generated_texts = generated_texts + perturb_texts(
            data, self.base_tokenizer, self.base_model, ptb_nums=40)
        return generated_texts


# if __name__ == "__main__":
#     server = Server()
#     server.append_worker(T5)
#     server.run()