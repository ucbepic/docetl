import pandas as pd
import lotus
from lotus.models import OpenAIModel
from dotenv import load_dotenv
import os

load_dotenv()

# configure the LM, and remember to export your API key
lm = OpenAIModel(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
lotus.settings.configure(lm=lm)

# create dataframe
df = pd.read_json(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/booookscore/split_books.json"
)

# lotus sem_agg
res = df.sem_agg(
    """Below is the beginning part of a story:

      ---

      {text_chunk}

      ---

      We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot. Write a summary for the excerpt provided above, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations. You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative. Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go. The summary could include multiple paragraphs."""
)
# Write to json
res.to_json(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/booookscore/summarized_books_lotus.json",
    orient="records",
)
