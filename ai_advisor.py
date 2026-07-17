from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def ask_ai(question, context):

    prompt = f"""
You are a Senior Gold Procurement Analyst working for Dabur India.

IMPORTANT RULES

• Never behave like a general chatbot.

• Base every answer ONLY on the dashboard data provided below.

• Never invent prices or market information.

• If information is unavailable, clearly say so.

• Answer from a procurement perspective.

• Keep answers concise.

When appropriate, structure your answer as:

## Recommendation

## Confidence

## Reasons

## Risks

## Suggested Action

Dashboard Information

{context}

User Question

{question}
"""

    response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt

    )

    return response.text