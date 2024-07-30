from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from code import ChatAngel
import os
import openai
from fastapi import FastAPI, Request
import uvicorn, json, datetime
from loguru import logger
import collections


id_2_name_file = {
    "Cupid": ("Cupid", "Cupid_1"),
    "Ben": ("Ben Smith", "Ben_Smith_2"),
    "Jacob": ("Jacob Anderson", "Jacob_Anderson_3"),
    "Ethan": ("Ethan Alexander Smith", "Ethan_Alexander_Smith_4"),
    "Julian": ("Julian Everett Hawthorne", "Julian_Everett_Hawthorne_5"),
    "Thomas": ("Thomas Anderson", "Thomas_Anderson_6"),
    "Ben_7": ("Benjamin David Johnson", "Benjamin_David_Johnson_7"),
    "Allison": ("Allison Johnson", "Allison_Johnson_8"),
    "Henry": ("Henry Thompson", "Henry_Thompson_9"),
    "Charles_10": ("Charles Edward Radcliffe", "Charles_Edward_Radcliffe_10"),
    "Jackson_11": ("Jackson Smith", "Jackson_Smith_11"),
    "Ethan_12": ("Ethan Taylor", "Ethan_Taylor_12"),
    "Liam_13": ("Liam Williams", "Liam_Williams_13"),
    "Mason_14": ("Mason Davis", "Mason_Davis_14"),
    "Aiden_15": ("Aiden Wilson", "Aiden_Wilson_15"),
    "Jack_16": ("Jack Hammer", "Jack_Hammer_16"),
    "Lucian_17": ("Lucian Steele", "Lucian_Steele_17"),
    "Oliver_18": ("Oliver Martinez", "Oliver_Martinez_18"),
    "Lucas_19": ("Lucas Johnson", "Lucas_Johnson_19"),
    "Michael_20": ("Michael Brown", "Michael_Brown_20"),
    "Liam_21": ("Liam Bennett", "Liam_Bennett_21"),
    "Jun_22": ("Jun Lee", "Jun_Lee_22"),
    "Caleb_23": ("Caleb Anderson", "Caleb_Anderson_23"),
    "Oliver_24": ("Oliver Martinez", "Oliver_Martinez_24"),
    "Andre_25": ("Andre Francisco", "Andre_Francisco_25"),
    "Alex_26": ("Alex Knight", "Alex_Knight_26"),
}

id_2_keywords = {
    'Ben': ['playful', 'photographer', 'poor', 'talented', 'witty', 'college student', 'quippy', 'young', 'gemini',
            'clever'],
    'Jacob': ['artistic', 'reading', 'aries', 'artist', 'talented', 'love animals', 'watching old movies', 'cooking',
              'eloquent', 'writing', 'baking', 'actor', 'guitar', 'expressive', 'young', 'rich'],
    'Ethan': ['confident', 'formula one', 'mix of control', 'direct', 'capricorn.', 'muscular', 'handsome',
              'domineering', 'horse racing', 'wealthy', 'good-looking', 'commanding', 'rich', 'soccer'],
    'Julian': ['poor', 'pianist', 'nostalgic', 'hockey', 'a subtle intensity', 'gemini', 'handsome', 'expressive',
               'jazz', 'young', 'good-looking', 'poetic'],
    'Thomas': ['english man', 'refined', 'confident', 'quiet', 'composed', 'leo', 'hunting', 'reserved', 'wealthy',
               'riding horses', 'rich'],
    'Ben_7': ['genuine', 'sincere', 'straightforward', 'a sense of vulnerability', 'young'],
    'Allison': ['journalist', 'articulate', 'concealing a genuine', 'charismatic', 'compassionate nature', 'humorous'],
    'Henry': ['razor-sharp', 'sardonic remarks', 'doctor', 'handsome', 'dry humor', 'good-looking'],
    'Charles_10': ['english man', 'mature', 'english duke'],
    'Jackson_11': ['eloquence', 'a sharp wit', 'barrister', 'laywer', 'handsome', 'mature', 'good-looking'],
    'Ethan_12': ['genuinely curious', 'college student', 'enthusiastic', 'muscular', 'innocent', 'playfully naive',
                 'young'],
    'Liam_13': ['articulate', 'linguistically masterful', 'eloquent', 'meticulous', 'precise', 'detective'],
    'Mason_14': ['folksy charm', 'football coach', 'wholesome', 'asian', 'young'],
    'Aiden_15': ['a hint of danger', 'spy', 'smooth', 'dangerous', 'mature', 'effortlessly confident', 'concise'],
    'Jack_16': ['english man', 'muscular', 'strong', 'mercenary', 'no-nonsense', 'mature', 'concise'],
    'Lucian_17': ['measured', 'deliberate', 'death', 'death god'],
    'Oliver_18': ['laced with sarcasm', 'disdain towards humanity', 'bitter', 'brooding', 'vampire', 'wild', 'vengeful',
                  'resentful tone', 'vampirea', 'muscular'],
    'Lucas_19': ['with a hint of mystery', 'self-sufficiency', 'killer', 'evasive', 'wild', 'muscular', 'violent',
                 'private', 'concise'],
    'Michael_20': ['unrefined vernacular', 'young', 'wild', 'rough'],
    'Liam_21': ['straight-forward', 'muscular', 'ex-husband'],
    'Jun_22': ['jealous', 'asian', 'husband', 'wealthy', 'rich'],
    'Caleb_23': ['dangerous', 'brilliant', 'cannibal', 'cultured'],
    'Oliver_24': ['husband', 'betrayed', 'angry'],
    'Andre_25': ['spain', 'husband', 'dangerous'],
    'Alex_26': ['cool', 'bodyguard']
}

query_send_words = ["give", "find", "send", "someone", "like", "I want", "love", "offer",
                    "boy", "friend", "boyfriend", "give me", "find me", "send me", "offer me"]

res_send_words = ["give you", "find you", "send you"]

keyword_2_ids = collections.defaultdict(list)

for user_id, keywords in id_2_keywords.items():
    for k in keywords:
        keyword_2_ids[k].append(user_id)


def get_role_first_news(user_id):
    role_name, _ = id_2_name_file[user_id]
    first_role_news_reply = [{"type": "text", "value": "<em>I stare at you with a mysterious and inscrutable expression~~</em>"}]

    role_first_news_path = "./data/role_first_news.txt"
    with open(role_first_news_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            role = line[0]

            if role == role_name:
                first_role_news_reply = [{"type": "text", "value": line[1].replace("(", "<em>").replace(")", "</em>")}]

    return "first_role_input_prompt", "first_role_model_res", first_role_news_reply


def get_map_first_news(place):
    map_chunk_dir = os.path.join("./data/chunks/Jacob_Anderson_3/", place)

    reply = []
    first_map_news_reply = {"type": "text", "value": "<em>I feel a bit blushing~~</em>"}
    first_map_default_reply = {"type": "default_rsp", "value": "It seems like you... have something to tell me? <em>my voice is a bit trembling~~</em>"}
    map_file = os.path.join(map_chunk_dir, "0.txt")
    with open(map_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 1:
                first_map_news_reply = {"type": "text", "value": line.strip()[16:-1].replace("(", "<em>").replace(")", "</em>")}
            elif i == 2:
                first_map_default_reply = {"type": "default_rsp", "value": line.strip()[10:-1].replace("(", "<em>").replace(")", "</em>")}
                break
            else:
                continue

    reply.append(first_map_news_reply)
    reply.append(first_map_default_reply)
    return "first_map_input_prompt", "first_map_model_res", reply


def get_default_reply(place, query):
    # map_chunk_dir = os.path.join("../../data/SQBench/chunks/Jacob_Anderson_3/", place)
    map_chunk_dir = os.path.join("./data/chunks/Jacob_Anderson_3/", place)
    file_list = sorted([k for k in os.listdir(map_chunk_dir)])

    flag = 0
    reply = []
    default_bot_reply = {"type": "text", "value": "I feel... not sure what to say~~"}
    default_human_reply = {}
    for file in file_list:
        map_file = os.path.join(map_chunk_dir, file)
        with open(map_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if query in line and flag == 0 and "warrensq:" in line:
                    flag = 1
                elif flag == 1 and "Jacob Anderson:" in line:
                    default_bot_reply = {"type": "text", "value": line[16:-1].replace("(", "<em>").replace(")", "</em>")}
                    reply.append(default_bot_reply)
                    flag = 2
                elif flag == 2 and "warrensq:" in line:
                    default_human_reply = {"type": "default_rsp", "value": line[10:-1].replace("(", "<em>").replace(")", "</em>")}
                    reply.append(default_human_reply)

                    return "default_reply_input_prompt", "default_reply_model_res", reply
                else:
                    continue

    reply.append(default_bot_reply)
    if len(default_human_reply) != 0:
        reply.append(default_human_reply)
    return "default_reply_input_prompt", "default_reply_model_res", reply


def conversation(user_id, chat_name, place, query, history, special_flag):
    query = query.replace("<em>", "(").replace("</em>", ")")
    history = [k.replace("<em>", "(").replace("</em>", ")") for k in history]

    # 非sq切换
    place = "common"
    special_flag = "none"

    if special_flag == "role_first_info":
        input_prompt, model_res, reply = get_role_first_news(user_id)
    elif special_flag == "map_first_info":
        input_prompt, model_res, reply = get_map_first_news(place)
    elif special_flag == "default_req":
        input_prompt, model_res, reply = get_default_reply(place, query)
    else:
        cut_limit = 1000
        input_prompt, input_ids = query_process(user_id, chat_name, place, query, history, cut_limit, special_flag)
        model_res = model_inference([input_ids])
        reply = reply_process(user_id, chat_name, query, model_res)

    return input_prompt, model_res, reply


def query_process(user_id, chat_name, place, query, histories, cut_limit=1000, special_flag="none", test=False):
    """
    用户query输入前处理

    input：
        user_id: agent id
        chat_name: 用户聊天名
        place: 地图名
        query: 用户query原始输入
        histories: 对话历史拼接
        cut_limit: 截断长度

    return：
        q: 处理后的模型输入prompt
        model_input_ids: prompttokenizer编码
    """
    # TODO p1: place -》 place_file，这版未更新place_index

    # pgc or ugc
    ugc_flag = 0
    if user_id not in id_2_name_file:
        ugc_flag = 1
        role_name, role_file = id_2_name_file["Alex_26"]
    else:
        role_name, role_file = id_2_name_file[user_id]

    output_dir = "./data/characters/{}".format(role_file)
    system_prompt_path = os.path.join(output_dir, "system_prompt.txt")
    db_folder = os.path.join(output_dir, "db_" + place.replace(" ", "_"))

    # chunk_dir
    story_text_folder = "./data/chunks/{}/{}/".format(role_file, place)
    chatbot = ChatAngel(system_prompt=system_prompt_path, story_text_folder=story_text_folder,
                        db_folder=db_folder, llm="Llama2GPT", embedding="bge_en")

    # query截断
    if len(query) > cut_limit:
        query = query[:cut_limit]
    content = chat_name + ":「" + query.replace("\n", " ").replace("\t", " ") + '」\n'

    # history用户昵称修正
    new_histories = []
    for k in histories:
        if role_name not in k and chat_name not in k:
            temp_list = k.split(":「")
            temp_list[0] = chat_name
            new_histories.append(":「".join(temp_list))
        else:
            new_histories.append(k)

    # ugc以Alex Knight为模板
    memory = chatbot.generate_prompt(content, histories).replace("warrensq", chat_name)

    if ugc_flag:
        memory = memory.replace("Alex Knight", user_id)

    system_prompt = (
        "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    )

    query_prompt = "[INST] {{query}} [/INST]"

    q = query_prompt.replace("{{query}}", system_prompt + memory)

    if test:
        return q
    else:
        model_input_ids = []
        model_input_ids += [tokenizer.bos_token_id]
        model_input_ids += tokenizer.encode(q, add_special_tokens=False)
        #     example += tokenizer.encode(r, add_special_tokens=False)
        #     example += [tokenizer.eos_token_id]

        return q, model_input_ids


def model_inference(input_ids):
    """
    模型inference结果

    input：
        input_ids: 模型输入prompt的tokenized input_ids

    return：
        model_res: 模型推理结果
    """
    # TODO p0: 推理需要batch化吗？

    inputs = torch.LongTensor(input_ids)
    inputs = inputs.to(device)
    prompt_length_list = [len(k) for k in inputs]

    gen_kwargs = {
        # "input_ids": inputs["input_ids"],
        "input_ids": inputs,
        "do_sample": True,
        "max_new_tokens": 512,
        "repetition_penalty": 1.2,
        "top_k": 0,
        "top_p": 0.8,
        #     "logits_processor": get_logits_processor()
    }

    generation_output = model.generate(**gen_kwargs)
    generation_output_list = generation_output.tolist()
    for i in range(len(generation_output_list)):
        generation_output_list[i] = generation_output_list[i][prompt_length_list[i]:]

    decoded_pred = tokenizer.batch_decode(generation_output_list, skip_special_tokens=True)
    model_res = decoded_pred[0]

    return model_res


def extract_move(text):
    """
    提取对话回复中的对话结果，标注出来

    input：
        text: 模型输出text

    return：
        sentences: 处理后的动作、对话分割列表
    """

    sentences = []
    stack = []
    current_sentence = ""
    for char in text:
        if char == '(':
            stack = ['(']
            if current_sentence:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        elif char == ')':
            if stack:
                stack.pop()
                if current_sentence:
                    sentences.append("<em>" + current_sentence.strip() + "</em>")
                    current_sentence = ""
            else:
                if current_sentence:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""

        else:
            current_sentence += char

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences


def show_split_sentence(reply_list, show_limit=80):
    """
    分割后列表展示，待更新

    input：
        text: extract_move分割列表

    return：
        sentences: 回传给server的对话格式
    """

    sentence_words = 0
    if len(" ".join(reply_list).split(" ")) <= show_limit:
        sentences = reply_list
    else:
        sentences = []
        for reply in reply_list:
            if len(sentences) != 0 and (sentence_words + len(reply.split(" "))) > show_limit:
                break
            else:
                sentences.append(reply)
                sentence_words += len(reply.split(" "))

    #     current_sentence = ""
    #     for reply in reply_list:
    #         if "<em>" in reply:
    #             if len(sentences) != 0 and (sentence_words + len(reply.split(" "))) > show_limit:
    #                 break
    #             else:
    #                 sentences.append(reply)
    #                 sentence_words += len(reply.split(" "))
    #         else:
    #             for char in reply:
    #                 if char in ['.', '!', '?']:
    #                     if current_sentence:
    #                         cut_flag = 1

    #                 elif cut_flag == 1:
    #                     if len(sentences) != 0 and (sentence_words + len(current_sentence.strip().split(" "))) > show_limit:
    #                         break
    #                     else:
    #                         sentences.append(current_sentence.strip())
    #                         sentence_words += len(current_sentence.strip().split(" "))

    #                     current_sentence = ""
    #                     cut_flag = 0

    #                 current_sentence += char

    #     if current_sentence[-1] in ['.', '!', '?'] and current_sentence.strip():
    #         sentences.append(current_sentence.strip())

    final_reply = " ".join(sentences).replace("*/", "").replace("{", "").replace("}", "").replace("\"", "").\
        replace("/", "").replace("[", "").replace("]", "").replace("→", "").replace(">>", "").replace("»", "").\
        replace("。", "").replace("，", "")

    sentences = [{"type": "text", "value": final_reply}]

    return sentences


def reply_process(user_id, chat_name, query, model_res):
    """
    模型生成回复后处理

    input：
        user_id: agent id
        chat_name: 用户聊天名
        query: 用户query原始输入
        model_res: 模型生成回复

    return：
        reply: 后处理回复，json格式 [{"type":"text","value":"aaa"}, {"type":"image","value":"bbb"},
        {"type":"video","value":"ccc"}, {"type":"card","value":"ddd"}]
    """
    # TODO p1: return错误测试后调整规则或者重跑 & 复杂度优化
    # TODO p0: reply过长分气泡一起处理

    # pgc or ugc
    if user_id not in id_2_name_file:
        role_name = user_id
    else:
        role_name, _ = id_2_name_file[user_id]

    # AI回复话术
    if "I'm just an AI" in model_res or "I'm just a chat" in model_res:
        return [{"type": "text", "value": "Em...That's a personal question, we'd better talk later~~"}]

    # 发送卡片规则判断
    send_flag = 0
    if "【SEND CARD】" in model_res:
        model_res = model_res.replace("【SEND CARD】", "").strip()
        send_flag = 1

    sign = ""
    if role_name + ":" in model_res:
        sign = role_name + ":"
    elif ":" in model_res:
        sign = ":"
    elif "：" in model_res:
        sign = "："

    if sign != "":
        reply_1 = [reply.strip() for reply in model_res.split(sign)][1]

        if chat_name + ":" in reply_1:
            reply_2 = [reply.strip() for reply in reply_1.split(chat_name + ":") if reply.strip() != ""][0]
        else:
            reply_2 = reply_1

        reply_3 = reply_2.replace("『", "").replace("「", "").replace("」", "").replace("』", ""). \
            replace("：", "").replace("（", "(").replace("）", ")")

        reply_4_list = extract_move(reply_3)

        replies = show_split_sentence(reply_4_list)

        # print("reply_1: {}".format(reply_1), flush=True)
        # print("reply_2: {}".format(reply_2), flush=True)
        # print("reply_3: {}".format(reply_3), flush=True)
        # print("reply_4_list: {}".format(reply_4_list), flush=True)
        # print("replies: {}".format(replies), flush=True)

        if len(replies[0]["value"].strip()) == 0:
            return [{"type": "text", "value": "<em>Feeling a bit excited</em>"}]

        # Cupid 下发人物策略
        if role_name == "Cupid":
            # query logic
            send_num = 0
            if send_flag != 1:
                for send_word in query_send_words:
                    if send_word in query:
                        send_num += 1
                    if send_num >= 2:
                        send_flag = 1
                        break

            # res logic
            if send_flag != 1:
                for send_word in res_send_words:
                    if send_word in model_res:
                        send_flag = 1
                        break

            # tag match
            if send_flag == 1:
                card_person = []

                for send_word, ids in keyword_2_ids.items():
                    if send_word in query or send_word in model_res:
                        card_person = ids
                        break

                replies.append({"type": "cardV2", "value": card_person})

        return replies

    else:
        # 人名未在回复中
        # return [{"text": "role name badcase note, contact warren to get log"}]
        return [{"type": "text", "value": "<em>Wondering what you mean by that.... Keeping silent~~</em>"}]


def interact(user_id, chat_name, place):
    """
    interact交互demo

    input：
        role_name: 选择聊天人物 名
        chat_name: 用户自定义人物昵称
        place: 聊天地图

    """

    record_dir = "./data/record/"
    os.makedirs(record_dir, exist_ok=True)
    file_indexs = [f.split(".")[0] for f in os.listdir(record_dir)]
    index = [str(i) for i in range(10000) if str(i) not in file_indexs][0]
    chat_file = "chat_" + index + ".txt"
    out = open(os.path.join(record_dir, chat_file), "w", encoding="utf-8")
    print(user_id + '\t' + place + '\n', flush=True, file=out)

    json_list = []
    histories = []
    # 聊天流程
    while 1:
        # try:

        # 用户输入query
        query = input(chat_name + '：')
        if query == 'quit':
            break

        # 生成回复
        prompt_text, input_ids = query_process(user_id, chat_name, place, query, histories)
        # print(tokenizer.decode(input_ids, skip_special_tokens=False), flush=True, file=out)
        model_res = model_inference([input_ids])
        replies = reply_process(user_id, chat_name, query, model_res)

        # 对话简要记录
        query_sent = chat_name + ":「" + query + '」'
        # reply_sents = [role_name + ":「" + reply[1] + '」' for reply in replies if reply[0] == "text"]

        # for reply_sent in reply_sents:
        #     print(reply_sent, flush=True)

        print(query_sent, flush=True)
        print(replies, flush=True)
        print(prompt_text, flush=True)
        print(flush=True)

        # print(query_sent, flush=True, file=out)
        # for reply_sent in reply_sents:
        #     print(reply_sent, flush=True, file=out)

        # 对话细节记录
        temp_dic = {
            "history": copy.deepcopy(histories),
            "query": query_sent,
            "reply": replies,
            "model_res": model_res,
            "prompt": tokenizer.decode(input_ids, skip_special_tokens=False),
        }
        json_list.append(temp_dic)

        # # 历史记录
        # histories.append(query_sent)
        # for reply_sent in reply_sents:
        #     histories.append(reply_sent)

        # except Exception as e:
        #     print("ERROR QUIT: {}", e)

    out.close()

    detail_file = "detail_" + index + ".json"
    out = open(os.path.join(record_dir, detail_file), "w", encoding="utf-8")
    print(json.dumps(json_list, indent=2, separators=(',', ': '), ensure_ascii=False), file=out)
    out.close()

    out.close()


app = FastAPI()


@app.get("/")
async def create_heart(request: Request):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": "",
        "status": 200,
        "time": time
    }
    return answer


@app.post("/chat")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    user_id = json_post_list.get('user_id')
    chat_name = json_post_list.get('chat_name')
    prompt = json_post_list.get('prompt')
    place = json_post_list.get('place')

    history = []
    if "history" in json_post_list:
        history = json_post_list.get('history')
    special_flag = "none"
    if "special_flag" in json_post_list:
        special_flag = json_post_list.get('special_flag')
        
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n\n[" + time + "] ")
    logger.info("\n" + str(json_post_list))

    input_prompt, model_res, reply = conversation(user_id, chat_name, place, prompt, history, special_flag)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": reply,
        "history": history,
        "status": 200,
        "time": time
    }

    logger.info("\n[" + time + "] ")
    logger.info("\n" + str([input_prompt]))
    logger.info("\n" + str(input_prompt))
    logger.info("\n" + str(model_res))
    logger.info("\n" + str(reply))
    logger.info("\n" + str("~~~~~~~~~~~~~~~~~~~~~~~~~~"))

    # log =  + '", prompt:"' + prompt + '", response:"' + repr(reply) + '"'
    # logger.info(log)
    #torch_gc()

    return answer


if __name__ == '__main__':
    # 非sq切换
    model_tokenizer_path = "./model/export_model_llama2_rolellm_sqllm_14/"
    # model_tokenizer_path = "./model/export_model_llama2_rolellm_sqllm_13/"

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path, trust_remote_code=True).half().to(device)
    model = model.eval()
    #tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    #model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    #model.eval()

    # 路径，每日分割时间，是否异步记录，日志是否序列化，编码格式，最长保存日志时间
    logger.add('./data/log/api.log', rotation='1 day', enqueue=True, serialize=False, encoding="utf-8", retention="10 days")
    logger.info("服务器重启！")

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)


# if __name__ == "__main__":
#     role_name = 'Jack Hammer'
#     query = 'hi'
#     model_res = "Jack Hammer:「😎」"
#
#     a = reply_process(role_name, query, model_res)
#     print(a, flush=True)


