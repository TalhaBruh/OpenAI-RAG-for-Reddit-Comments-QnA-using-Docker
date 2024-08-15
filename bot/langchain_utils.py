import config
import reddit_utils
import tiktoken
import openai
import logging
import llm_utils
from typing import List
from datetime import datetime
import pandas as pd
import pinecone
import database
from uuid import uuid4

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores import Pinecone

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='app.log', level=logging.DEBUG)

query_text = f"(Todays Date: {datetime.now().strftime('%Y-%b-%d')}) Revise and summarize"\
            " the article in 250 words or less by incorporating relevant information from the comments."\
            " Ensure the content is clear, engaging, and easy to understand for a"\
            " general audience. Avoid technical language, present facts objectively,"\
            " and summarize key comments from Reddit. Ensure that the overall"\
            " sentiment expressed in the comments is accurately reflected. Optimize"\
            " for highly original content. Don't be trolled by joke comments. Ensure"\
            " its written professionally, in a way that is appropriate for the"\
            " situation. Format the document using markdown and include links from the"\
            " original article/reddit thread."

settings =  {
        "system_role": "You are a helpful assistant.",
        "query": query_text,
        "chunk_token_length": 1000,
        "max_number_of_summaries": 1,
        "max_token_length": 4097,
        "selected_model": "text-davinci-003",
        "selected_model_type": "OpenAI Instruct"
    }

# setup openai
openai.api_key = config.openai_api_key
if config.openai_api_base is not None:
    openai.api_base = config.openai_api_base

pinecone.init(
    api_key= config.pinecone_api_key,
    environment= config.pinecone_env
)

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 60.0,
}


class RedditGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=openai.api_key
        )
        self.index_name = 'reddit-chatbot-telegram'
        self.vectorstore=None
        if self.index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=self.index_name,
                metric='dotproduct',
                dimension=1536  # 1536 dim of text-embedding-ada-002
            )
            
        self.text_field = 'text'
        self.index = pinecone.Index(self.index_name)
        # self.vectorstore = None
        self.vectorstore = Pinecone(
            pinecone.Index(self.index_name), self.embed.embed_query, self.text_field
        )

    async def send_qa_response(self, message, user_id, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try: 
                if self.model in {"gpt-3.5-turbo"}:
                    # chat completion llm
                    llm = ChatOpenAI(
                        openai_api_key=openai.api_key ,
                        model_name=self.model,
                        temperature=0.5
                    )
                    # conversational memory
                    conversational_memory = ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        k=1,
                        return_messages=True
                    )
                    # retrieval qa chain
                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=self.vectorstore.as_retriever()
                    )
                    qa.retriever.search_kwargs = {
                        "filter": {"user_id": {"$eq": user_id}}
                    }
                    answer = qa.run(message)
                    n_input_tokens, n_output_tokens = 0,0

                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message(self, message, user_id, dialog_messages=[], chat_mode="assistant", is_url = False):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo"}:
                    if is_url: #If reddit url
                        messages = self._generate_prompt_for_thread(message, chat_mode, user_id)
                    else:
                        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)

                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    def _generate_prompt_for_thread(self, url, chat_mode, user_id):
        reddit_data = reddit_utils.get_reddit_praw(url)

        prompts: List[str] = []
        pinecone.delete_index(self.index_name)
        pinecone.create_index(
            name=self.index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

        title, selftext, subreddit, comments = (
            reddit_data["title"],
            reddit_data["selftext"],
            reddit_data["subreddit"],
            reddit_data["comments"],
        )

        if not comments:
            comments = "No Comments"

        #NOTE: Setting chunk token length to 2000 for now
        groups = llm_utils.group_bodies_into_chunks(comments, 2000)

        if len(groups) == 0:
            groups = ["No Comments"]

        if (selftext is None) or (len(selftext) == 0):
            selftext = "No selftext"
        
        #NOTE: Setting chunk token length to 2000 for now
        groups = (
            llm_utils.group_bodies_into_chunks(comments, 2000)
            if len(groups) > 0
            else ["No Comments"]
        )

        init_prompt = f"{title}\n{selftext}"

        system_role, query, max_tokens = (
            settings["system_role"],
            settings["query"],
            settings["max_token_length"],
        )

        for i, comment_group in enumerate(groups[:settings["max_number_of_summaries"]]):
            complete_prompt = (
                f"{query}\n\n"
                + "```"
                + f"Title: {init_prompt}\n\n"
                + f'<Comments subreddit="r/{subreddit}">\n{comment_group}\n</Comments>\n'
                + "```"
            )

            prompts.append(complete_prompt)

        #################################################################################3
        grouped_data = llm_utils.group_bodies_into_chunks(complete_prompt, 2000)
        data = pd.DataFrame(grouped_data, columns=['context'])
        data['name'] = subreddit
        data['user_id'] = user_id
        data['uuid'] = [uuid4() for _ in range(len(data.index))]
        data.drop_duplicates(subset='context', keep='first', inplace=True)

        self.index = pinecone.GRPCIndex(self.index_name)

        # Reset index and ensure 'index' column is added
        data = data.reset_index(drop=True)
        data = data.reset_index()
        batch_size = 100

        for i in range(0, len(data), batch_size):
            # get end of batch
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]

            # first get metadata fields for this record
            metadatas = [{
            self.text_field : record[1],  # 'text' will contain the same data as 'context'
            'name': record[2],
            'user_id': record[3]
            } for record in batch.itertuples(index=False)]

            # print(metadatas)
            # get the list of contexts / documents
            documents = batch['context'].tolist()

            # create document embeddings
            embeds = self.embed.embed_documents(documents)

            # get IDs and convert them to strings
            ids = batch['uuid'].astype(str).tolist()

            # add everything to pinecone
            self.index.upsert(vectors=list(zip(ids, embeds, metadatas)))

        # switch back to normal index for langchain
        self.index = pinecone.Index(self.index_name)

        self.vectorstore = Pinecone(
            self.index, self.embed.embed_query, self.text_field
        )

        ##################################################################################


        return [{"role": "system", "content": complete_prompt}]

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]

async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())
