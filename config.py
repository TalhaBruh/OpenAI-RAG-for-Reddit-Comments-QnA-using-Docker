import dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s - %(lineno)d")

try:
    config_env = dotenv.dotenv_values(".env")
except Exception as e:
    logging.error("An error occurred while loading .env file : %s", str(e))


openai_api_key = config_env['OPENAI_API_KEY']
pinecone_api_key = config_env['PINECONE_API_KEY']
pinecone_env= config_env['PINECONE_ENV']
reddit_client_id = config_env['REDDIT_CLIENT_ID']
reddit_client_secret = config_env['REDDIT_CLIENT_SECRET']
reddit_username = config_env['REDDIT_USERNAME']
reddit_password = config_env['REDDIT_PASSWORD']
reddit_user_agent = config_env['REDDIT_USER_AGENT']