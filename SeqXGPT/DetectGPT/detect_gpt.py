import httpx
import msgpack
import json
import random

from tqdm import tqdm


def access_sniffer(text, api_url, language="en", get_data=0, get_dc=0):
    """
    language = "en" or "cn"
    get_data = 0 or 1 or 2
    get_dc = 0 or 1, 2, 3
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "language": language,
            "get_data": get_data,
            "get_dc": get_dc,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        print(prediction)
        content = None
    return content

def access_api(text, api_url, do_generate=False):
    """
    :param text:        input text
    :param api_url:     api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


# TODO: you need to set the following parameter.
#   in_file, out_file
#   t5_url, url

if __name__ == "__main__":
    in_file = "your_data_path/en_gptj_lines.jsonl"
    out_file = "your_data_path/perturb_en_gptj_lines.jsonl"
    
    with open(in_file, 'r') as f:
        samples_test = [json.loads(line) for line in f]

    detect_gpt_samples = []
    processed_sentence = 0

    idx = -1
    for item in tqdm(samples_test):
        idx += 1
        text = item['text']
        label = item['label']

        t5_url = ""     # url of t5 inference server
        url = ""        # url of particular model inference server
        ptb_num = 40

        discard_num = 0
        try:
            ptb_texts = access_api(text, t5_url)
            if ptb_texts is None or len(ptb_texts) < ptb_num:
                print("Probably perturb OOD! {} samples have been discarded.".format(discard_num))
                discard_num += 1
                continue
            
            losses = access_api(ptb_texts, url)
            if losses is None or len(losses) < ptb_num:
                print("Probably perplexity OOD! {} samples have been discarded.".format(discard_num))
                discard_num += 1
                continue

            golden_losses = access_api([text], url)
            if golden_losses is None or len(golden_losses) < 1:
                print("Probably perplexity OOD! {} samples have been discarded.".format(1))
                discard_num += 1
                continue
            
            item['losses'] = losses
            item['golden_loss'] = golden_losses[0]
            detect_gpt_samples.append(item)
            processed_sentence += 1
        except Exception as e:
            print(e)
            print("fail to process this sample, discard it")
            print(idx)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(detect_gpt_samples, f)

    print("processed_sentence:", str(processed_sentence))