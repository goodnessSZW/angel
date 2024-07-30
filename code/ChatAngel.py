import os
import sys
from transformers import AutoTokenizer
from .ChromaDB import ChromaDB
from .utils import luotuo_openai_embedding, tiktokenizer
from .utils import response_postprocess


class ChatAngel:
    def __init__(self, system_prompt=None, story_text_folder=None, db_folder=None, llm='openai',
                 embedding='bge_en', max_len_story=1500, max_len_history=1200, verbose=False):
        super(ChatAngel, self).__init__()
        self.verbose = verbose

        # constants
        self.k_search = 5
        self.max_len_story = max_len_story
        self.max_len_history = max_len_history
        self.story_prefix_prompt = "\nClassic scenes for the role are as follows:"
        self.dialogue_divide_token = '\n###\n'
        self.dialogue_bra_token = '「'
        self.dialogue_ket_token = '」'

        # load system_prompt
        if system_prompt:
            self.system_prompt = self.check_system_prompt(system_prompt)

        # load LLM
        if llm == "Llama2GPT":
            self.llm, self.tokenizer = self.get_models('Llama2GPT')
        else:
            print(f'warning! undefined llm {llm}, use openai instead.')
            self.llm, self.tokenizer = self.get_models('openai')

        # load RAG model
        if embedding == 'bge_en':
            from .utils import get_bge_embedding
            self.embedding = get_bge_embedding
        else:
            print(f'warning! undefined embedding {embedding}, use luotuo_openai instead.')
            self.embedding = luotuo_openai_embedding
        
        # load RAG database
        if story_text_folder:
            if not db_folder:
                db_folder = "db_" + story_text_folder.split("/")[-1].replace(" ", "_")   
            else:
                print("Input to db_folder: {}".format(db_folder), flush=True)
                
            if not os.path.exists(db_folder):
                self.db = self.build_story_db(story_text_folder, db_folder)
            else:
                self.db = ChromaDB()
                self.db.load(db_folder)
        else:
            self.db = None
            print('warning! database not yet figured out, both story_db and story_text_folder are not inputted.')

    def check_system_prompt(self, system_prompt):
        # if system_prompt end with .txt, read the file with utf-8
        # else, return the string directly
        if system_prompt.endswith('.txt'):
            with open(system_prompt, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return system_prompt

    def get_models(self, model_name):
        if model_name == "Llama2GPT":
            from .Llama2GPT import Llama2GPT, Llama_tokenizer
            return (Llama2GPT(), Llama_tokenizer)
        else:
            print(f'warning! undefined model {model_name}, use openai instead.')
            sys.exit()

    def build_story_db(self, text_folder, db_folder):
        # 实现读取文本文件夹,抽取向量的逻辑
        db = ChromaDB()

        strs = []

        # scan all txt file from text_folder
        for file in os.listdir(text_folder):
            # if file name end with txt
            if file.endswith(".txt"):
                file_path = os.path.join(text_folder, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    strs.append(f.read())

        if self.verbose:
            print(f'starting extract embedding... for { len(strs) } files')

        vecs = []
        for mystr in strs:
            vec = self.embedding(mystr)
            vecs.append(vec)
            # print(len(vec))

        db.init_from_docs(vecs, strs, db_folder)

        return db

    def add_story_with_expire(self, query):
        if self.db is None:
            print("No vec DB！")
            return

        query_vec = self.embedding(query)
        stories = self.db.search(query_vec, self.k_search)
        story_string = self.story_prefix_prompt + self.dialogue_divide_token
        sum_story_token = self.tokenizer(story_string)

        for story in stories:

            # # 推理是注释掉
            # if query.strip() in story.strip():
            #     continue

            story_token = self.tokenizer(story.strip()) + self.tokenizer(self.dialogue_divide_token)
            if sum_story_token + story_token > self.max_len_story:
                break
            else:
                sum_story_token += story_token
                story_string += story.strip() + self.dialogue_divide_token

        return story_string

    def generate_prompt(self, query, history):
        message = ""
        message += self.system_prompt + "\n"

        story_string = self.add_story_with_expire(query)
        message += story_string

        history_message = self.add_history(history)
        if history_message != "":
            message += history_message

        message += query

        # self.llm.user_message(target)

        return message

    def add_history(self, history_list):

        if len(history_list) == 0:
            return ""

        sum_history_token = 0
        flag = 0
        for history in history_list:
            current_count = 0
            if history is not None:
                current_count += self.tokenizer(history)

            sum_history_token += current_count

            # 长度截断 & 数据库读取历史对话是否为最新版
            if sum_history_token > self.max_len_history or "「" not in history:
                break
            else:
                flag += 1

        history_message = ""
        for history in history_list[-flag:]:
            if ":「」" in history:
                continue
            else:
                history_message += history.replace("\n", " ").replace("\t", " ")
                history_message += "\n"

        return history_message

    def chat(self, text, role):
        # add system prompt
        self.llm.initialize_message()
        self.llm.system_message(self.system_prompt)

        # add story
        query = self.get_query_string(text, role)
        self.add_story(query)

        # add history
        self.add_history()

        # add query
        self.llm.user_message(query)

        # get response
        response_raw = self.llm.get_response()

        response = response_postprocess(response_raw, self.dialogue_bra_token, self.dialogue_ket_token)

        # record dialogue history
        self.dialogue_history.append((query, response))

        return response