import reddit_utils
import llm_utils
from uuid import uuid4
import pinecone
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import config
import pandas as pd


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

def generate_prompt_for_thread(url):
    reddit_data = reddit_utils.get_reddit_praw(url)

    prompts: List[str] = []

    pinecone.init(
        api_key= config.pinecone_api_key,
        environment= config.pinecone_env
    )
    index_name = "reddit-chatgpt-db"

    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(
        name=index_name,
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
    # data['user_id'] = user_id
    data['uuid'] = [uuid4() for _ in range(len(data.index))]
    data.drop_duplicates(subset='context', keep='first', inplace=True)

    index = pinecone.GRPCIndex(index_name)

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
        'text' : record[1],  # 'text' will contain the same data as 'context'
        'name': record[2]
        # 'user_id': record[3]
        } for record in batch.itertuples(index=False)]

        # print(metadatas)
        # get the list of contexts / documents
        documents = batch['context'].tolist()

        embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=config.openai_api_key
        )
        # create document embeddings
        embeds = embed.embed_documents(documents)

        # get IDs and convert them to strings
        ids = batch['uuid'].astype(str).tolist()

        # add everything to pinecone
        index.upsert(vectors=list(zip(ids, embeds, metadatas)))

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, 'text'
    )

    ##################################################################################


    return [{"role": "system", "content": complete_prompt}], vectorstore
