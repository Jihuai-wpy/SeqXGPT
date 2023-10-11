import os
import argparse

from mosec import Server
from collections import OrderedDict
from backend_model import (SnifferGPTNeoModel, 
                           SnifferGPT2Model, 
                           SnifferGPTJModel, 
                           SnifferWenZhongModel, 
                           SnifferSkyWorkModel, 
                           SnifferDaMoModel, 
                           SnifferLlamaModel, 
                           SnifferChatGLMModel, 
                           SnifferAlpacaModel, 
                           SnifferDollyModel,
                           SnifferStableLMRawModel,
                           SnifferStableLMTunedModel)
from backend_t5 import T5

MODEL_MAPPING_NAMES = OrderedDict([
    ("gpt2", SnifferGPT2Model),
    ("gptneo", SnifferGPTNeoModel),
    ("gptj", SnifferGPTJModel),
    ("llama", SnifferLlamaModel),
    ("wenzhong", SnifferWenZhongModel),
    ("skywork", SnifferSkyWorkModel),
    ("damo", SnifferDaMoModel),
    ("chatglm", SnifferChatGLMModel),
    ("alpaca", SnifferAlpacaModel),
    ("dolly", SnifferDollyModel),
    ("stablelm_raw", SnifferStableLMRawModel),
    ("stablelm_tuned", SnifferStableLMTunedModel)
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="gpt2",
        help=
        "The model to use. You can choose one of [gpt2, gptneo, gptj, llama, wenzhong, skywork, damo, chatglm, alpaca, dolly].",
    )
    parser.add_argument("--gpu",
                        type=str,
                        required=False,
                        default='0',
                        help="Set os.environ['CUDA_VISIBLE_DEVICES'].")

    parser.add_argument("--port", help="mosec args.")
    parser.add_argument("--timeout", help="mosec args.")
    parser.add_argument("--debug", action="store_true", help="mosec args.")
    return parser.parse_args()


if __name__ == "__main__":
    # --model: [damo, gpt2, gptj, gptneo, wenzhong, skywork, llama]
    # python backend_api.py --port 6006 --timeout 30000 --debug --model=damo --gpu=3
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.model == 't5':
        server = Server()
        server.append_worker(T5)
        server.run()
    else:
        sniffer_model = MODEL_MAPPING_NAMES[args.model]
        server = Server()
        server.append_worker(sniffer_model)
        server.run()