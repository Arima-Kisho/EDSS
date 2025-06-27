from statistics import mean
from torch.utils.data import Dataset
import openai  # For GPT-3 API ...
import os
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime
import spacy
from scipy.stats import entropy
from sentence_transformers import util
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(1)

    # enter your OpenAI key below
    openai.api_key = ""

    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    elif args.model == "turbo":
        engine = "gpt-3.5-turbo"
    else:
        raise ValueError("model is not properly defined ...")

    max_attempts = 9
    attempt = 1
    success = False
    while attempt <= max_attempts and not success:
        try:
            if engine == "gpt-3.5-turbo":

                if args.method == "role_play":
                    conversation = [
                        {"role": "user", "content": args.role_setting},
                        {"role": "assistant", "content": args.reply},
                        {"role": "user", "content": input}]
                else:
                    conversation = [{"role": "user", "content": input}]

                # print(conversation)
                # exit()
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=conversation,
                    temperature=0,
                    max_tokens=512
                )
            else:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=input,
                    max_tokens=512,
                    temperature=0,
                    stop=None
                )

            success = True
            # print(response)
            # exit()

        except Exception or openai.error.RateLimitError or openai.error.APIError or openai.error.ServiceUnavailableError:
            attempt += 1

            time.sleep(1.05)
    if not success:
        raise Exception("overload")

    if engine == "gpt-3.5-turbo":
        return response['choices'][0]['message']['content']
    else:
        return response["choices"][0]["text"]


class Decoder():
    def __init__(self, args):
        print_now()

    def decode(self, args, input):
        response = decoder_for_gpt3(args, input)
        return response


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("bigbench_date", "object_tracking"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A', 'B', 'C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = choice_index[i]
                        # a = key
                q = q + " " + choice
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader


# ver 0.2
def answer_cleansing(args, pred):
    print("pred_before : " + pred)

    if "few_shot" in args.method:
    # if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        match = re.search(r'boxed{(yes|no)}', pred)

        if match:
            # 返回第一个匹配到的 'yes' 或 'no'
            pred = [match.group(1)]  # 返回 ['yes'] 或 ['no']
        else:
          pred = pred.split(" ")
          pred = [i for i in pred if i in ("yes", "no")]    
        print("pred: " , pred) 
    elif args.dataset == "last_letters":
        right_index = pred.rfind('"')
        if right_index != -1:
            left_index = pred[:right_index].rfind('"')
            pred = pred[left_index:right_index + 1].lower()
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if "few_shot" in args.method:
        # if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ["zero_shot_cot", "ps+", "ef_cot", "ef_cot_roberta", "ef_cot_deberta", "es_cot"]:
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    print("pred_after : " + pred)

    return pred
    
def create_auto_demo_text(args, demo_file, cot_flag):
    x, z, y = [], [], []

    with open(demo_file, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))

    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []

    # example sentences ...
    if args.dataset in ("multiarith", "gsm8k", "addsub", "singleeq", "svamp"):
        x.append(
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append(
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append(
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")

        x.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")

        x.append(
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")

        x.append(
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append(
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")

        x.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")
    elif args.dataset == "aqua":
        x.append(
            "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64")
        z.append(
            "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be50. The answer is (A).")
        y.append("A")

        x.append(
            "If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append(
            "If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")

        x.append(
            "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append(
            "The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. ")
        y.append("E")

        x.append(
            "How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append(
            "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.")
        y.append("B")
    elif args.dataset == "last_letters":
        x.append(
            "Take the last letters of the words in \"Elon Musk\" and concatenate them.")
        z.append(
            "The last letter of \"Elon\" is \"n\". The last letter of \"Musk\" is \"k\".Concatenating them is \"nk\".")
        y.append("nk")
        x.append(
            "Take the last letters of the words in \"Larry Page\" and concatenate them.Concatenating them is \"ye\".")
        z.append(
            "The last letter of \"Larry\" is \"y\". The last letter of \"Page\" is \"e\". Concatenating them is \"ye\".")
        y.append("ye")
        x.append(
            "Take the last letters of the words in \"Sergey Brin\" and concatenate them.")
        z.append(
            "The last letter of \"Sergey\" is \"y\". The last letter of \"Brin\" is \"n\". Concatenating them is \"yn\".")
        y.append("yn")
        x.append(
            "Take the last letters of the words in \"Bill Gates\" and concatenate them.")
        z.append(
            "The last letter of \"Bill\" is \"l\". The last letter of \"Gates\" is \"s\". Concatenating them is \"ls\".")
        y.append("ls")
    elif args.dataset == "coin_flip":
        x.append(
            "A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. ")
        y.append("yes")

        x.append(
            "A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. ")
        y.append("yes")

        x.append(
            "A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. ")
        y.append("no")

        x.append(
            "A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. ")
        y.append("no")

        x.append(
            "A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.")
        y.append("no")

        x.append(
            "A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.")
        y.append("no")

        x.append(
            "A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up.")
        y.append("yes")

        x.append(
            "A coin is heads up. Ryan flips the coin. Shaunda flips the coin. Is the coin still heads up?")
        z.append(
            "The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. ")
        y.append("yes")
    elif args.dataset == "object_tracking":
        x.append(
            " ")
        z.append(
            "")
        y.append("")
    elif args.dataset == "bigbench_date":
        x.append(
            "2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?")
        z.append(
            "If 2015 is coming in 36 hours, then it is coming in 2 days. 2 days before 01/01/2015 is 12/30/2014, so today is 12/30/2014. So one week from today will be 01/05/2015.")
        y.append("01/05/2015")

        x.append(
            "The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?")
        z.append(
            "If the first day of 2019 was Tuesday, then 01/01/2019 was a Tuesday. Today is the first monday, would be six days later. So today is 01/07/2019.")
        y.append("01/07/2019")

        x.append(
            "The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?")
        z.append(
            "One day after 06/01/1943 is 06/02/1943, so today is 06/02/1943. 10 days before today is 05/23/1943.")
        y.append("05/23/1943")

        x.append(
            "It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 04/19/1969. 24 hours later is one day after today, which would be 04/20/1969.")
        y.append("04/20/1969")

        x.append(
            "Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 03/12/2002. So the date 24 hours later will be 03/13/2002.")
        y.append("03/13/2002")

        x.append(
            "Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the dateyesterday in MM/DD/YYYY?")
        z.append(
            "The last day of February is the 28th, so Jane was born on 02/28/2001. Today is her 16-year old birthday, so today is 02/28/2017. So yesterday was 02/27/2017")
        y.append("02/28/2017")

    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text


def postprocess_output(pred, input=None):
    pred = ''.join(map(str, pred))
    if input is not None:
        # remove the original input in the output if have
        if input in pred:
            pred = pred[len(input):]

    # for Self-RAG
    pred = pred.replace("</s>", "")

    # for Phi
    # if "<|im_start|>" in pred:
    if "<|im_sep|>" in pred:
        # idx = pred.rfind("<|im_start|>assistant")
        idx = pred.rfind("<|assistant<|im_sep|>")
        if idx != -1:
            pred = pred[idx + len("<|assistant<|im_sep|>"):]
    pred = pred.replace("<|im_end|>", "")
    if "[INST]" in pred:
        idx = pred.rfind("[INST]")
        if idx != -1:
            pred = pred[idx + len("[INST]"):]
    pred = pred.replace("[/INST]", "")

    # for TinyLlama
    if "<|assistant|>" in pred:
        idx = pred.rfind("<|assistant|>")
        if idx != -1:
            pred = pred[idx + len("<|assistant|>"):]
        pred = pred.replace("<|user|>", "")
        pred = pred.replace("<|assistant|>", "")
        pred = pred.replace("<|system|>", "")

    # for Llama and Mistral
    if "[/INST]" in pred:
        idx = pred.rfind("Answer:[/INST]")
        if idx != -1:
            pred = pred[idx + len("Answer:[/INST]"):]

    pred = pred.replace("\n", " ")
    # remove extra white space
    pred = " ".join(pred.split())

    return pred


# ver 0.3
def complexity_score(sentence, weights=(1.0, 1.0, 1.0, 1.0, 1.0)):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    tree_depth, sub_clauses, conjunctions = calculate_syntactic_complexity(doc)
    entropy_score = calculate_information_entropy(sentence)
    vector_density = calculate_word_vector_density(sentence)

    # 标准化并组合成综合评分
    features = np.array([tree_depth, sub_clauses, conjunctions, entropy_score, vector_density])
    normalized_features = (features - np.mean(features)) / np.std(features)
    score = np.dot(normalized_features, weights)
    return score


def calculate_syntactic_complexity(doc):
    max_depth = max([token.dep_.count("sub") for token in doc])
    sub_clauses = sum(1 for token in doc if token.dep_ in {"advcl", "ccomp", "xcomp", "acl"})
    conjunctions = sum(1 for token in doc if token.dep_ == "conj")
    return max_depth, sub_clauses, conjunctions


def calculate_information_entropy(sentence):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([sentence])
    tfidf_scores = X.toarray()[0]
    return entropy(tfidf_scores)


def calculate_word_vector_density(sentence):
    w2v_model = KeyedVectors.load_word2vec_format("../Models/GoogleNews-vectors-negative300.bin", binary=True)
    words = sentence.split()
    vectors = [w2v_model[word] for word in words if word in w2v_model]
    if len(vectors) == 0:
        return 0
    mean_vector = np.mean(vectors, axis=0)
    variances = np.var(vectors, axis=0)
    return np.mean(variances)


def save_complexities_and_threshold(complexities, threshold, dataset="aqua"):
    """将复杂度分数和阈值保存到 JSON 文件"""
    filename = dataset + "-" + "complexity_scores.json"
    data = {"complexities": complexities, "complex_threshold": threshold}
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_complexities_and_threshold(dataset="aqua"):
    """从 JSON 文件加载复杂度分数和阈值"""
    filename = dataset + "-" + "complexity_scores.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get("complexities", {}), data.get("complex_threshold")
    return {}, None


def split_question(text):
    # 去掉开头和结尾的空格
    text = text.strip()

    # 使用正则表达式分隔句子，并捕获最后一个句子
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|,)\s', text)
    # sentences = re.split(r'[.,?!]\s*', text.strip())

    if len(sentences) > 1:
        # 将最后一个句子作为问题部分，其余部分作为信息部分
        information = ' '.join(sentences[:-1]).strip()
        question = sentences[-1].strip()
        return information, question
    else:
        return text, ""  # 如果没有分隔符，返回原文本和 None

