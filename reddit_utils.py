import re
import praw  # type: ignore
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple
from data_types import RedditData, GenerateSettings
import logging
import validators
import config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='app.log', level=logging.DEBUG)


def is_reddit_url(url : str) -> bool :
    """
    Check if the URL is a valid Reddit URL.
    """
    if validators.url(url):
        pattern = re.compile(
            r"^"  # Start of the string
            r"(http(s)?:\/\/)?"  # Optional "http://" or "https://"
            r"(www\.)?"  # Optional "www."
            r"reddit\.com\/"  # "reddit.com/"
            r"([a-zA-Z0-9-_]+\/)+"  # 1 or more letters, numbers, -'s, or _ followed by "/"
            r"[a-zA-Z0-9-_]+"  # 1 or more letters, numbers, -'s, or _'s
        )
        return bool(pattern.match(url))
    return False


def replace_last_token_with_json(reddit_url: str) -> str:
    """Replace the last token in the Reddit URL with '.json'."""
    tokens = reddit_url.rsplit("/", 1)
    return f"{tokens[0]}.json"

def format_date(timestamp: float) -> str:
    """Format a timestamp into a human-readable date."""
    date: datetime = datetime.fromtimestamp(timestamp)
    return date.strftime("%Y-%b-%d %H:%M")

def get_comments(comment: Any, level: int = 0) -> str:
    """Get the comments from a Reddit thread."""
    result = ""

    author_name = comment.author.name if comment.author else "[deleted]"
    created_date = format_date(comment.created_utc)

    result += f"{created_date} [{author_name}] {comment.body}\n"

    for reply in sorted(
        comment.replies, key=lambda reply: reply.created_utc, reverse=True
    ):
        result += "    " * level
        result += "> " + get_comments(reply, level + 1)

    return result


def get_reddit_praw(
    reddit_url: str
) -> RedditData:
    """
    Process the reddit thread JSON and generate a summary.
    """
    # json_url = replace_last_token_with_json(reddit_url)
    json_url = reddit_url

    try:
        # Get the subreddit and metadata from the JSON
        match = re.search(r"/r/(\w+)/", json_url)
        if match:
            subreddit = match.group(1)
        else:
            logging.error("No subreddit found in URL")
            raise ValueError("No subreddit found in URL")


        reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            password=config.reddit_password,
            user_agent=config.reddit_user_agent,
            username=config.reddit_username
        )

        submission: Any = reddit.submission(url=json_url)  # type: ignore
        submission.comment_sort = "top"  # sort comments by score (upvotes - downvotes)
        submission.comments.replace_more(limit=None)

        title: Optional[str] = submission.title
        selftext: Optional[str] = submission.selftext

        if not title:
            logging.error("No title found in JSON")
            raise ValueError("No title found in JSON")

        comment_string = ""
        for comment in submission.comments:
            comment_string += get_comments(comment)

        return RedditData(
            title=title, selftext=selftext, subreddit=subreddit, comments=comment_string
        )

    except Exception as ex:  # pylint: disable=broad-except
        logging.error(f"Error getting reddit meta data: {ex}")
