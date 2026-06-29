import os

os.environ['HF_HOME'] = f'/mount/arbeitsdaten/tcl/tclext/golubaa/thesis/.cache'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import load_data
from lookup import prompt_lookup

if __name__ == "__main__":
    task = CLF
    target = GALTAN
    model_name = 'allenai/Olmo-3-7B-Instruct'
    random_seed = 7
    temperature = 0.85  # default 0.6
    top_p = 0.85  # default 0.95
    max_new_tokens = 30
    num_return_sequences = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    # load model
    quantization_config = BitsAndBytesConfig(  # to decrease memory consumption
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization_config
    ).to(device)

    # read data
    test_df, test_dataloader = load_data(
        task=task,
        target=target,
        train_or_test='test',
        mount=mount_no_backup,
        test_batch_size=16,
        random_seed=random_seed)

    prompt = prompt_lookup[target]

    # run test
    responses = []
    try:
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            messages = [
                [{"role": "user", "content": prompt.format(sent=sent)}]
                for sent in batch['texts']
            ]
            tokenized_chat = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True
            ).to(device)

            output = model.generate(
                **tokenized_chat,
                do_sample=True,  # default True
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id  # otherwise gives a warning
            )

            batch_resp = tokenizer.batch_decode(output, skip_special_tokens=True)
            for i, sent in enumerate(batch['texts']):
                for j in range(num_return_sequences):
                    responses.append([sent, batch_resp[i * num_return_sequences + j]])

            del batch, messages, tokenized_chat, output, batch_resp
    except:
        pass

    # save responses
    preds_path = get_preds_path_llm(task, target, model_name,
                                    temperature, top_p, max_new_tokens, num_return_sequences)
    pd.DataFrame(responses).to_csv(preds_path)
