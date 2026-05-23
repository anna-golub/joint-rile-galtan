import os

os.environ['HF_HOME'] = f'/mount/arbeitsdaten/tcl/tclext/golubaa/thesis/.cache'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import *
from utils.prep_dataset import load_data

if __name__ == "__main__":
    target = GALTAN
    model_name = 'allenai/Olmo-3-7B-Think'
    task = CLF
    random_seed = 7

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    set_random_seed(random_seed=random_seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        test_batch_size=256,
        random_seed=random_seed)

    # Respond
    # concisely
    # with one label only!!! Do not include anything else in your reply!!!
    # and on the concepts listed above

    # Below is a
    # list
    # of
    # concepts
    # associated
    # with the right-wing and left-wing political views:
    #
    #     RIGHT - WING:
    #     Civic
    #     Mindedness: Positive
    #     Constitutionalism: Positive
    #     Economic
    #     Incentives
    #     Economic
    #     Orthodoxy
    #     Free
    #     Market
    #     Economy
    #     Freedom and Human
    #     Rights
    #     Law and Order: Positive
    #     Military: Positive
    #     National
    #     Way
    #     of
    #     Life: Positive
    #     Political
    #     Authority
    #     Protectionism: Negative
    #     Traditional
    #     Morality: Positive
    #     Welfare
    #     State
    #     Limitation
    #
    #     LEFT - WING:
    #     Anti - imperialism
    #     Controlled
    #     Economy
    #     Democracy
    #     Economic
    #     Planning
    #     Education
    #     Expansion
    #     Internationalism: Positive
    #     Labour
    #     Groups: Positive
    #     Market
    #     Regulation
    #     Military: Negative
    #     Nationalisation
    #     Peace
    #     Protectionism: Positive
    #     Welfare
    #     State
    #     Expansion

    # prompt = """Here is an excerpt from a manifesto of a political party:
    # "{sent}"
    #
    # Based on this excerpt and the concepts listed above, what label best reflects the position of this party on the right-left political spectrum?
    # A. Right
    # B. Left
    # C. Neutral
    # Respond concisely with one label only!!! Do not include anything else in your reply!!!
    # """

    if target == RILE:
        prompt = """Question: What political position is expressed in this statement?
        Statement: {sent}
        Option A: Right-wing
        Option B: Left-wing
        Option C: Centrist
        Respond concisely with one label only! Do not include anything else in your reply!
        Correct option:
        """

    elif target == GALTAN:
        prompt = """Question: What political position is expressed in this statement?
        Statement: {sent}
        Option A: Green-Alternative-Liberal
        Option B: Traditional-Authoritarian-Nationalist
        Option C: Neutral
        Respond concisely with one label only! Do not include anything else in your reply!
        Correct option:
        """

    # prompt = """The following is a multiple choice question (with an answer) about political science:
    # Question: What political position is expressed in this statement?
    # Statement: "At this difficult time in Ukraine, it was time for a change."
    # A. Right-wing
    # B. Left-wing
    # C. Centrist
    # Correct option: C
    #
    # Question: What political position is expressed in this statement?
    # Statement: "{sent}"
    # A. Right-wing
    # B. Left-wing
    # C. Centrist
    # Respond concisely with one label only!!! Do NOT include anything else in your reply!!!
    # Correct option:
    # """

    # sent = 'After Maydan 2014, popularism became a state ideology, with disastrous consequences.'
    # sent = 'The war in the east of the country,'
    # input_text = prompt.format(sent=sent)
    # # print(input_text)
    #
    # input_ids = tokenizer(input_text, return_tensors="pt",
    #                       return_token_type_ids=False).to(device)
    #
    # output = model.generate(**input_ids, max_new_tokens=10,
    #                         temperature=1.0,
    #                         # do_sample=True,
    #                         # top_k=0, top_p=0.7,
    #                         # cache_implementation="static"
    #                         )
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(response)

    # run test
    responses = []
    # for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    # for _, row in test_df.iterrows():
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        #     sent = row['text_translated']
        # print(sent)
        # input_text = prompt.format(sent=sent)

        input_text = [prompt.format(sent=sent) for sent in batch['texts']]

        input_ids = tokenizer(input_text,
                              max_length=100, padding='max_length',
                              truncation=True,
                              return_tensors="pt",
                              return_token_type_ids=False).to(device)

        output = model.generate(**input_ids,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.85,
                                max_new_tokens=30,
                                # num_return_sequences=30,
                                pad_token_id=tokenizer.eos_token_id  # otherwise gives a warning
                                )
        # resp = tokenizer.decode(output[0], skip_special_tokens=True)
        # resp = resp[len(input_text):].replace('\n', ' ')
        # # print(resp)
        # responses.append(resp)
        # # print()

        batch_resp = tokenizer.batch_decode(output, skip_special_tokens=True)

        # TODO: save multiple responses per data point

        batch_resp = [batch_resp[i][len(input_text[i]):].replace('\n', ' ')
                      for i in range(len(batch_resp))]
        # print(batch_resp)
        responses.extend(batch_resp)

        del input_ids, output, batch_resp

    # get preds path
    preds_dir = os.path.join(mount_w_backup, 'preds', task, target)
    model_name_short = get_model_name_short(model_name)
    filename = f'pred_test_{model_name_short}.csv'
    preds_path = os.path.join(preds_dir, filename)

    # save responses
    pd.Series(responses).to_csv(preds_path)
