PROMPT_DICT_1Step = {
    "aqua": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
    "gsm8k": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
    "singleeq": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
    "multiarith": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the most core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
    "addsub": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the most core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
    "svamp": (
        "From now on, you are an expert in information structures, and you can always analyze the information of entities and understand the differences from natural language problems.\n"
        "Based on the information given, complete the following operations.\n"
        "1. Extract the most core entities.(Hint: Don't show explanations)\n"
        "2. Clarify the relationship between entities, and check the value of the entity.\n"
        "given information: {context_num}\n"
    ),
}
PROMPT_DICT_2Step = {
    "aqua": (
        "Here's the information extracted from the entities and relationships: {information}.\n"
        "Restate \"{context}\" based on this information to make it clearer and more straightforward, and remove complex or implicit descriptions.Ensure original precision of value.\n"
    ),
    "addsub": (
        "Here's the information extracted from the entities and relationships: {information}.\n"
        "Based on the entity information, restate \"{context}\", to make it clearer and more straightforward, and remove complex or implicit descriptions.Ensure original precision of value.\n"
    ),
    "multiarith": (
        "Here's the entity information: {information}.\n"
        "Based on the entity information, restate \"{context}\", to make it clearer and more straightforward, and remove complex or implicit descriptions.Ensure original precision of value.\n"
    ),
    "svamp": (
        "Here's the information extracted from the entities and relationships: {information}.\n"
        "Based on the entity information, restate \"{context}\", to make it clearer and more straightforward, and remove complex or implicit descriptions.Ensure original precision of value.\n"
    ),
    "gsm8k": (
        "Here's the information extracted from the entities and relationships: {information}.\n"
        "Please use this information to reorganize \"{context}\" into a concise description as a new description.\n"
    ),
    "singleeq": (
        "Here's the information extracted from the entities and relationships: {information}.\n"
        "Restate \"{context}\" based on this information to make it clearer and more straightforward, and remove complex or implicit descriptions.Ensure original precision of value.\n"
    ),
}
PROMPT_DICT_3Step = {
    "addsub": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step patiently without think more(Ensure original precision of value).\n"
    ),
    "aqua": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step without think more(Ensure original precision of value).\n"
    ),
    "svamp": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step without think more(Ensure original precision of value).\n"
    ),
    "singleeq": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step without think more(Ensure original precision of value).\n"
    ),
    "multiarith": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step patiently with think more(Ensure original precision of value).\n"
    ),
    "gsm8k": (
        "Q: {question}. In simple terms: {context} {core}\n"
        "A: Solve the problem step by step patiently with think more(Ensure original precision of value).\n"
    ),
}



def get_prompt_name(args):
    prompt_name = f"{args.dataset}L"
    return prompt_name

"1. Extract the core entities.(Hint: Don't show explanations)\n"
"2. Clarify the relationship between entities.\n"
"3. Clarify the value of each entity.\n"