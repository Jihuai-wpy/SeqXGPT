import torch
import transformers
import numpy as np

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc, CharLevelTokenizerPPLCalc, SPChatGLMTokenizerPPLCalc
from backend_utils import split_sentence
# mosec
from mosec import Worker
from mosec.mixin import MsgpackMixin
# llama
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class SnifferBaseModel(MsgpackMixin, Worker):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = None
        self.base_model = None
        self.generate_len = 512

    def forward_calc_ppl(self):
        pass

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            tokenized = self.base_tokenizer(self.text, return_tensors="pt").to(
                self.device)
            tokenized = tokenized.input_ids
            gen_tokens = self.base_model.generate(tokenized,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.squeeze()
            result = self.base_tokenizer.decode(gen_tokens.tolist())
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return gen_texts

    def forward(self, data):
        """
        :param data: ['text': str, "do_generate": bool]
        :return:
        """
        self.text = data["text"]
        self.do_generate = data["do_generate"]
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()


class SnifferGPT2Model(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'gpt2-xl')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'gpt2-xl')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

class SnifferGPTNeoModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-neo-2.7B')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-neo-2.7B', device_map="auto", load_in_8bit=True)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferGPTJModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-j-6B')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-j-6B', device_map="auto", load_in_8bit=True)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferLlamaModel(SnifferBaseModel):
    """
    More details can be seen:
        https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaModel
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = ''
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.unk_token_id = self.base_tokenizer.unk_token_id
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True)
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferWenZhongModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        # bpe tokenizer
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese',
            device_map="auto",
            load_in_8bit=True)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferSkyWorkModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'SkyWork/SkyTextTiny', trust_remote_code=True)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'SkyWork/SkyTextTiny', device_map="auto", load_in_8bit=True)
        all_special_tokens = self.base_tokenizer.all_special_tokens
        self.ppl_calculator = CharLevelTokenizerPPLCalc(
            all_special_tokens, self.base_model, self.base_tokenizer,
            self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferDaMoModel(SnifferBaseModel):

    def __init__(self):
        from modelscope.models.nlp import DistributedGPT3
        from modelscope.preprocessors import TextGenerationJiebaPreprocessor
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_dir = ''
        self.base_tokenizer = TextGenerationJiebaPreprocessor(model_dir)
        self.base_model = DistributedGPT3(model_dir=model_dir, rank=0)
        self.base_model.to(self.device)
        self.all_special_tokens = ['']

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits.squeeze()  # seq-len, V
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))  # [seq-len] ?
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def calc_token_ppl(self, input_ids, ll):
        input_ids = input_ids[0].cpu().tolist()
        # char-level
        words = split_sentence(self.text)
        chars_to_words = []
        for idx, word in enumerate(words):
            char_list = list(word)
            chars_to_words.extend([idx for i in range(len(char_list))])

        # get char_level ll
        tokenized_tokens = []
        for input_id in input_ids:
            tokenized_tokens.append(self.base_tokenizer.decode([input_id]))
        chars_ll = []
        # because tokenizer don't include <s> before the sub_tokens,
        # so the first sub_token's ll cannot be obtained.
        token = tokenized_tokens[0]
        if token in self.all_special_tokens:
            char_list = [token]
        else:
            char_list = list(token)
        chars_ll.extend([0 for i in range(len(char_list))])
        # next we process the following sequence
        for idx, token in enumerate(tokenized_tokens[1:]):
            if token in self.all_special_tokens:
                char_list = [token]
            else:
                char_list = list(token)
            chars_ll.extend(ll[idx] for i in range(len(char_list)))

        # get token_level ll
        start = 0
        ll_tokens = []
        while start < len(chars_to_words) and start < len(chars_ll):
            end = start + 1
            while end < len(chars_to_words
                            ) and chars_to_words[end] == chars_to_words[start]:
                end += 1
            if end > len(chars_ll):
                break
            ll_token = chars_ll[start:end]
            ll_tokens.append(np.mean(ll_token))
            start = end

        # get begin_word_idx
        begin_token = self.base_tokenizer.decode([input_ids[0]])
        if begin_token in self.all_special_tokens:
            char_list = [begin_token]
        else:
            char_list = list(begin_token)
        begin_word_idx = chars_to_words[len(char_list) - 1] + 1

        return ll_tokens, begin_word_idx

    def forward_calc_ppl(self):
        # bugfix: clear the self.inference_params, so we can both generate and calc ppl
        self.base_model.train()
        self.base_model.eval()
        input_ids = self.base_tokenizer(self.text)['input_ids'].to(self.device)
        labels = input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        outputs = self.base_model(tokens=input_ids,
                                  labels=input_ids,
                                  prompts_len=torch.tensor([input_ids.size(1)
                                                            ]))

        loss, ll = self.calc_sent_ppl(outputs, labels)
        ll_tokens, begin_word_idx = self.calc_token_ppl(input_ids, ll)
        return [loss, begin_word_idx, ll_tokens]

    def forward_gen(self):
        # 1. single generate
        if isinstance(self.text, str):
            input_ids = self.base_tokenizer(self.text)['input_ids'].to(
                self.device)
            gen_tokens = self.base_model.generate(input_ids,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.sequences
            gen_tokens = gen_tokens[0].cpu().numpy().tolist()
            result = self.base_tokenizer.decode(gen_tokens)
            return result
        # 2. batch generate
        # damo model didn't implement batch_encode and batch_decode, so we use a for loop here
        elif isinstance(self.text, tuple):
            batch_res = []
            for text in self.text:
                input_ids = self.base_tokenizer(text)['input_ids'].to(
                    self.device)
                gen_tokens = self.base_model.generate(
                    input_ids, do_sample=True, max_length=self.generate_len)
                gen_tokens = gen_tokens.sequences
                gen_tokens = gen_tokens[0].cpu().numpy().tolist()
                result = self.base_tokenizer.decode(gen_tokens)
                batch_res.append(result)
            return batch_res


class SnifferChatGLMModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True)
        self.base_model = transformers.AutoModel.from_pretrained(
            "THUDM/chatglm-6b",
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.ppl_calculator = SPChatGLMTokenizerPPLCalc(
            self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        # self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_new_tokens=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(
                gen_tokens.tolist(), skip_special_tokens=True)
            result = gen_texts[0]
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            self.text = list(self.text)
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(
                **inputs, do_sample=True, max_new_tokens=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(
                gen_tokens.tolist(), skip_special_tokens=True)
            return gen_texts
        

class SnifferAlpacaModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = LlamaTokenizer.from_pretrained(
            "chavinlo/alpaca-native", trust_remote_code=True)
        self.base_model = LlamaForCausalLM.from_pretrained(
            "chavinlo/alpaca-native",
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True)
        # self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'

        PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                """
        # instruction = "Rewrite the following paragraph in a different style using your own words."

        # treat single generate as batch generate
        if isinstance(self.text, str):
            self.text = [self.text]

        processed_text = [PROMPT_FORMAT.format(instruction=text) for text in self.text]
        inputs = self.base_tokenizer(processed_text,
                                     padding=True,
                                     return_tensors="pt").to(self.device)
        gen_tokens = self.base_model.generate(
            **inputs, do_sample=True, max_new_tokens=self.generate_len)
        gen_texts = self.base_tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=True)
        # TODO change gen_text.find() to gen_text.split() 
        # TODO output.gen_text("### Response:")[1].strip()
        gen_texts = [gen_text[gen_text.find('### Response:\n') + 14 : ].strip() for gen_text in gen_texts]
        
        if len(gen_texts) == 1:
            return gen_texts[0]
        else:
            return gen_texts


class SnifferDollyModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_model = AutoModelForCausalLM.from_pretrained('databricks/dolly-v1-6b', trust_remote_code=True, 
                                                               device_map="auto", load_in_8bit=True)
        self.base_tokenizer = AutoTokenizer.from_pretrained('databricks/dolly-v1-6b', padding_side="left")

    def forward_gen(self):
        # NOTE only use for rephrase
        PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                        ### Instruction:
                        {instruction}

                        ### Response:
                        """
        # instruction = "Rewrite the following paragraph in a different style using your own words."
        response_key_token_id = self.base_tokenizer.encode("### Response:")[0]
        end_key_token_id = self.base_tokenizer.encode("### End")[0]

        # treat single generate as batch generate
        if isinstance(self.text, str):
            self.text = [self.text]
        
        processed_text = [PROMPT_FORMAT.format(instruction=text) for text in self.text]
        # inputs = self.base_tokenizer(processed_text, padding=True, max_length=512, truncation=True,return_tensors="pt").to(self.device)
        inputs = self.base_tokenizer(processed_text, return_tensors="pt").to(self.device)
        gen_tokens = self.base_model.generate(**inputs, pad_token_id=self.base_tokenizer.pad_token_id, 
                                              eos_token_id=end_key_token_id, do_sample=True, max_new_tokens=1024, top_p=0.92, top_k=0)
        gen_texts = []
        discard_num = 0
        for tokens in gen_tokens:
            tokens = tokens.cpu()
            response_positions = np.where(tokens == response_key_token_id)[0]
            # becatuse we truncate the sequences to max_length=512, simply discard these samples
            if len(response_positions) > 0:
                response_pos = response_positions[0]
                end_pos = None
                end_positions = np.where(tokens == end_key_token_id)[0]
                if len(end_positions) > 0:
                    end_pos = end_positions[0]

                print("eos_pos: {}".format(end_pos))
                gen_texts.append(self.base_tokenizer.decode(tokens[response_pos + 1 : end_pos]).strip())
            else:
                discard_num += 1
        print("discard_num: {}/{}".format(discard_num, len(gen_tokens)))

        if len(gen_texts) == 1:
            return gen_texts[0]
        else:
            return gen_texts
    

class SnifferStableLMRawModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-base-alpha-7b", trust_remote_code=True, 
                                                               device_map="auto", load_in_8bit=True)
        self.base_tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-7b", padding_side="left")
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # <|USER|><|ASSISTANT|><|SYSTEM|><|padding|><|endoftext|>
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class SnifferStableLMTunedModel(SnifferBaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b", trust_remote_code=True, 
                                                               device_map="auto", load_in_8bit=True)
        self.base_tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b", padding_side="left")

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
            - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
            - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
            - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
            - StableLM will refuse to participate in anything that could harm a human.
            """

        # treat single generate as batch generate
        if isinstance(self.text, str):
            self.text = [self.text]

        processed_text = [f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>" for user_prompt in self.text]
        inputs = self.base_tokenizer(processed_text, padding=True, return_tensors="pt").to("cuda")
        gen_tokens = self.base_model.generate(**inputs, max_new_tokens=self.generate_len, temperature=0.7, 
                                         do_sample=True, stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
        
        gen_texts = []
        for tokens in gen_tokens:
            tokens = tokens.cpu()
            assistant_key_token_id = 50279
            response_pos = np.where(tokens == assistant_key_token_id)[0][-1]
            gen_texts.append(self.base_tokenizer.decode(tokens[response_pos + 1 :], skip_special_tokens=True))

        if len(gen_texts) == 1:
            return gen_texts[0]
        else:
            return gen_texts