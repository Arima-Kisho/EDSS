import argparse
from tqdm import tqdm
from datetime import date

from prompt import *
from utils import *
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, util
import jsonlines
from collections import Counter



# 实体间的因果关系
# 实体纠错

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...
    # decoder = Decoder(args)
    model = load_model(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_manual_cot":
        demo = create_demo_text(args, cot_flag=True)
    elif args.method == "few_shot_auto_cot":
        demo_file = f"./demos/{args.dataset}/{args.model}/demo.json"
        demo = create_auto_demo_text(args, demo_file=demo_file, cot_flag=True)
    else:
        pass

    total = 0
    correct_list = []
    base_error_file = f"log/{args.dataset}/{args.model}/{args.method}/error.jsonl"

    for i, data in enumerate(dataloader):

        print('*************************')
        print("{}st data".format(i + 1))

        # Prepare question template ...
        x, y = data
        item = {}
        prompt_name = get_prompt_name(args)
        x = x[0]
        item["question"] = x
        if args.dataset in ["aqua", "bigbench_date", "object_tracking", "commonsensqa"]:
            parts = x.split("Answer Choices:")
            x = parts[0].strip()  # 问题部分
            c = parts[1].strip()  # 选项部分
            item["choices"] = c
        context, core = split_question(x)
        y = y[0].strip()
        item["core"] = core
        item["context"] = context
        item["time"] = date.today().strftime("%Y-%m-%d")
        sentences = [sentence.strip() for sentence in
                     re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|,)\s', context) if sentence]
        numbered_sentences = [f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences)]
        qs = [q.strip() for q in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|,)\s', x) if q]
        numbered_qs = [f"{i + 1}. {q}" for i, q in enumerate(qs)]

        # -1 is zero-cot; 0 is few-cot; 1 is ours; 
        complexity = 0
        if args.method in ["ef_cot_roberta", "ef_cot_deberta"]:
            complexity = markByBert(x)
        print("complexity: ", complexity, "\n")
        if complexity == -1:
            if args.dataset in ["aqua", "bigbench_date", "object_tracking", "commonsensqa"]:
                x = x + c
            x = "Q: " + x + "\n" + "A:"
            x = x + " " + args.cot_trigger
            z = generate_response(x, model)
            z = z[0]
            z2 = x + z + " " + args.direct_answer_trigger
            pred = generate_response(z2, model)
            print(z2, pred[0])
        elif complexity == 0:
            if args.dataset in ["aqua", "bigbench_date", "object_tracking", "commonsensqa"]:
                x = x + c
            x = "Q: " + x + "\n" + "A:"
            x = demo + x
            z = generate_response(x, model)
            z = postprocess_output(z)
            # z = z[0].split("\n\n")[0] 私下用
            pred = z
            print(x, pred)
        elif complexity == 1:
            item["context_num"] = numbered_sentences
            prompt_name = args.dataset
            formatted_prompt = PROMPT_DICT_1Step[prompt_name].format_map(item)
            z = generate_response(formatted_prompt, model)
            z = postprocess_output(z[0])
            if i == 0:
                print(formatted_prompt)
            print("z: ", z)

            item["information"] = z
            prompt_name_2 = args.dataset
            for k in range(1):
                formatted_prompt = PROMPT_DICT_2Step[prompt_name_2].format_map(item)
                z2 = generate_response(formatted_prompt, model)
                z2 = postprocess_output(z2[0])
                if i == 0:
                    print(formatted_prompt)
                print("z2: ", z2)
                item["information"] = z2

            prompt_name_3 = args.dataset
            temperature = 0.7 if args.SC else 0.0
            n = 10 if args.SC else 1
            formatted_prompt = PROMPT_DICT_3Step[prompt_name_3].format_map(item)
            if args.SC:
                answer_list = []
                for i in range(n):
                    z3 = generate_response(formatted_prompt, model, temperature)
                    z3 = postprocess_output(z3[0])

                    if args.dataset in ["aqua", "bigbench_date", "object_tracking", "commonsensqa"]:
                        x = x + c
                    pred = x + "\n" + z3 + args.direct_answer_trigger
                    pred = generate_response(pred, model)
                    try:
                        pred_answer = postprocess_output(pred)
                    except:
                        pred_answer = None

                    # Clensing of predicted answer ...
                    pred_answer = answer_cleansing(args, pred_answer)
                    answer_list.append(pred_answer)
                    print(pred_answer)
                collection_words = Counter(answer_list)
                pred = collection_words.most_common(1)[0][0]
            else:
                z3 = generate_response(formatted_prompt, model, temperature)
                z3 = postprocess_output(z3[0])
                if i == 0:
                    print(formatted_prompt)
                print("z3: ", z3)

                if args.dataset in ["aqua", "bigbench_date", "object_tracking", "commonsensqa"]:
                    x = x + c
                pred = x + "\n" + z3 + args.direct_answer_trigger
                pred = generate_response(pred, model)
                print(pred)
        pred = pred[0] if isinstance(pred, list) else pred

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)

        # Choose the most frequent answer from the list ...
        # print("pred : {}".format(pred))
        print("pred:", pred)
        print("GT : " + y)
        print('*************************')

        # We clean the pred and GT here!
        pred = clean_pred(pred)
        y = clean_ans(y)

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        # if abs(pred-y) <= 1e-9:
        #     total+=1
        #     print(f"exist float error. pred is {pred} label is {y}")
        correct_list.append(correct)
        total += 1  # np.array([y]).size(0)

        if not correct:
            with jsonlines.open(base_error_file, mode="a") as error_writer:
                error_log = {
                    "error_question": x,
                    "error_response": z,  # 错误的回答过程
                    "error_answer": pred
                }
                error_writer.write(error_log)

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    print_now()


def load_model(args):
    model = LLM(model=f"../Models/{args.model}",
                tensor_parallel_size=args.world_size,
                trust_remote_code=True,
                seed=args.random_seed,
                max_model_len=7040  
                )
    return model


def call_model(prompts, model, temperature=0.0, max_new_tokens=256):
    stop = None
    # phi-3
    stop = ['<|endoftext|>']
    # llama-3.1
    # stop = ['<|eom_id|>', '<|eot_id|>', '<|end_of_text|>']
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_new_tokens, stop=stop)
    preds = model.generate(prompts, sampling_params)

    preds = [pred.outputs[0].text for pred in preds]

    return preds


def generate_response(formatted_prompt, model=None, temperature=0.0):
    text = call_model([formatted_prompt], model=model, temperature=temperature)

    return text

def messages_to_string(messages):
    """Convert messages to a single string for the model."""
    formatted = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"
    return formatted.strip()


def clean_ans(ans):
    new_ans = ""
    for i in range(len(ans)):
        if ans[i] == ",":
            continue
        new_ans += ans[i]
    # print(ans, new_ans)

    if '.' in new_ans:
        pos = new_ans.find('.')
        if len(new_ans) - pos - 1 > 7:
            new_ans = new_ans[:pos + 7]
    return new_ans


def clean_pred(pred):
    if '.' in pred:
        pred = pred.rstrip('0')
        if pred.endswith('.'):
            pred = pred[:-1]

    if '.' in pred:
        pos = pred.find('.')
        if len(pred) - pos - 1 > 7:
            pred = pred[:pos + 7]
    return pred


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # parser.add_argument(
    #     "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    # )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--SC", default=False, type=bool, help="self consistency"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    # parser.add_argument(
    #     "--model", type=str, default="gpt3", choices=["turbo","gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    # )
    parser.add_argument(
        "--model", type=str, default="gpt3",
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot",
        choices=["zero_shot_cot", "ps+", "ef_cot", "ef_cot_roberta", "ef_cot_deberta", "few_shot_manual_cot",
                 "few_shot_auto_cot", "es_cot"], help="method"
    )
    # parser.add_argument(
    #     "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    # )
    # parser.add_argument(
    #     "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    # )
    # parser.add_argument(
    #     "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    # )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    # parser.add_argument(
    #     "--api_time_interval", type=float, default=1.0, help=""
    # )
    # parser.add_argument(
    #     "--log_dir", type=str, default="./log/", help="log directory"
    # )
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size to use multiple GPUs.")

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals, retain original precision of value) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals, retain original precision of value) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals, retain original precision of value) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals, retain original precision of value) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals, retain original precision of value) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the final answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    # args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.method in ["zero_shot_cot"]:
        args.cot_trigger = "Let's think step by step."
    if args.method in ["es_cot"]:
        args.cot_trigger = "Solve the problem step by step without think more(Ensure original precision of value)."
    elif args.method == "ps+":
        args.cot_trigger = "Let's first understand the problem, extract relevant variables and their corresponding numerals, and make a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."
    # elif args.cot_trigger_no == 2:
    #     args.cot_trigger = "We should think about this step by step."
    # elif args.cot_trigger_no == 3:
    #     args.cot_trigger = "First,"
    # elif args.cot_trigger_no == 4:
    #     args.cot_trigger = "Before we dive into the answer,"
    # elif args.cot_trigger_no == 5:
    #     args.cot_trigger = "Proof followed by the answer."
    # elif args.cot_trigger_no == 6:
    #     args.cot_trigger = "Let's think step by step in a realistic way."
    # elif args.cot_trigger_no == 7:
    #     args.cot_trigger = "Let's think step by step using common sense and knowledge."
    # elif args.cot_trigger_no == 8:
    #     args.cot_trigger = "Let's think like a detective step by step."
    # elif args.cot_trigger_no == 9:
    #     args.cot_trigger = "Let's think about this logically."
    # elif args.cot_trigger_no == 10:
    #     args.cot_trigger = "Let's think step by step. First,"
    # elif args.cot_trigger_no == 11:
    #     args.cot_trigger = "Let's think"
    # elif args.cot_trigger_no == 12:
    #     args.cot_trigger = "Let's solve this problem by splitting it into steps."
    # elif args.cot_trigger_no == 13:
    #     args.cot_trigger = "The answer is after the proof."
    # elif args.cot_trigger_no == 14:
    #     args.cot_trigger = "Let's be realistic and think step by step."
    # else:
    #     raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
