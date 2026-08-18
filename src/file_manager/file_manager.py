from langchain.chat_models import init_chat_model
from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv
from langchain.tools import tool
from typing import TypedDict
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import asyncio

load_dotenv()

model = init_chat_model(
    model="ollama:llama3.2",
    temperature=0.5
)

parse_client = AsyncLlamaCloud()

async def parse_api(file_name):
    file = await parse_client.files.create(file=f"./files/{file_name}", purpose="parse")
    result = await parse_client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    return "\n\n".join(
    page.markdown
    for page in result.markdown.pages
)


class MetaData(BaseModel):
    file_name: str = Field(
        description="Populate this field with the name of the file as it is stored."
    )

    file_description: str = Field(
        description="Populate this field with a short and concise description of the file contents."
    )

    tags: list[str] = Field(
        description="Populate this field with a list of important keywords from the contents of the given file."
    )

class State(TypedDict):
    file_name: str
    parsed_markdown: str
    file_meta_data: MetaData

meta_data_extractor = model.with_structured_output(MetaData)

async def extract_markdown(state: State):
    """Extract given file contents into a markdown."""

    file_markdown = await parse_api(state["file_name"])

    return {
        "parsed_markdown": file_markdown
    }

def extract_metadata(state: State):
    """Given a markdown extract metadata from it."""

    metadata = meta_data_extractor.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant. Given the name and markdown of a specific file, extract and return metadata from it."
            ),
            HumanMessage(
                content=f"File name: {state['file_name']}\nFile markdown: {state['parsed_markdown']}"
            )
        ]
    )

    return {
        "file_meta_data": metadata
    }



graph = StateGraph(State)

graph.add_node("extract_markdown", extract_markdown)
graph.add_node("extract_metadata", extract_metadata)

graph.add_edge(START, "extract_markdown")
graph.add_edge("extract_markdown", "extract_metadata")
graph.add_edge("extract_metadata", END)

agent = graph.compile()


async def main():

    stream = await agent.astream_events(
        {
            "file_name": "Global_warming.pdf"
        },
        version="v3"
    )

    async def consume_messages():
        async for message in stream.messages:
            print(message.node, flush=True)

            async for token in message.text:
                print(token, end="", flush=True)

    await asyncio.gather(consume_messages())


asyncio.run(main())
