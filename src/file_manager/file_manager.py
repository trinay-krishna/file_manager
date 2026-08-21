from langchain.chat_models import init_chat_model
from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv
from langchain.tools import tool
from typing import TypedDict
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import asyncio
from typing import Annotated
import operator
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import sys

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
        description="The exact name of the file as it is stored, including its file extension."
    )

    file_description: str = Field(
        description="A short, concise description of the file's main content, subject, and purpose. Focus on what the document is about rather than summarizing every detail."
    )

    tags: list[str] = Field(
        description="A list of important keywords that describe the file's subject, topics, entities, or document type. Use concise, meaningful terms that would help identify or search for this file later."
    )


class Shelf_Metadata(BaseModel):
    shelf_name: str = Field(
        description="A short, clear name that represents the category or subject of documents that belong on this shelf. The name should be broad enough to contain multiple related files."
    )

    shelf_description: str = Field(
        description="A short, concise description of the type of documents this shelf is intended to contain. Describe the common subject, purpose, or category shared by files that belong on this shelf."
    )

    tags: list[str] = Field(
        description="A list of important keywords representing the subjects, categories, document types, or themes associated with this shelf. Use broad, reusable terms that can be matched against file metadata."
    )



class File(BaseModel):
    metadata: MetaData


class Shelf(BaseModel):
    metadata: Shelf_Metadata = Field(
        description="Metadata that describes the category, subject, and type of files that belong on this shelf."
    )

    files: list[File] = Field(
        description="The files currently stored on this shelf."
    )


class CheckShelf(BaseModel):
    belongs_to_shelf: bool = Field(
        description="Whether the given file is an appropriate fit for this shelf. Return true only when the file's subject, purpose, and content are sufficiently aligned with the shelf's metadata."
    )

    confidence: float = Field(
        description="A confidence score from 0.0 to 1.0 indicating how certain you are that the file belongs to this shelf. Use 1.0 for an extremely strong match and 0.0 for no meaningful match."
    )

    reason: str = Field(
        description="A brief explanation of why the file does or does not belong to this shelf. Compare the file's description and tags with the shelf's description and tags."
    )

    shelf: Shelf | None = Field(
        description="The shelf being evaluated. Return the provided shelf when evaluating it, or null when no shelf is being evaluated."
    )

class ShelfMatch(BaseModel):
    shelf: Shelf 
    result: CheckShelf

class State(TypedDict):
    file_name: str
    parsed_markdown: str
    file_meta_data: MetaData
    file: File
    shelves: list[Shelf]
    shelf_matches: Annotated[list[ShelfMatch], operator.add]
    assigned_shelf: Shelf | None

class WorkerState(TypedDict):
    shelf_matches: Annotated[list[ShelfMatch], operator.add]
    shelf: Shelf
    file: File

#Added for the sake of testing while developing, should ideally be fetched from database.
shelves = [
    #Shelf(files=[], metadata=Shelf_Metadata(shelf_name="Environmental Studies", shelf_description="# Shelf for Environmental Studies and Sustainability", tags=["environmental studies", "sustainability", "climate change", "global warming", "ecology", "conservation"]))
]

meta_data_extractor = model.with_structured_output(MetaData)
shelf_assign_model = model.with_structured_output(CheckShelf)
shelf_creator = model.with_structured_output(Shelf_Metadata)

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

def create_file(state: State):
    """Creates a File object consisting the given metadata"""

    file = File(metadata=state["file_meta_data"])

    return {
        "file": file
    }

def fetch_shelves(state: State):
    """Fetch all shelves from database"""

    return {
        "shelves": shelves
    }

def check_valid_shelf(state: WorkerState):
    """Decide if the given file belongs to the given shelf or not"""

    file_metadata = state["file"].metadata
    shelf_metadata = state["shelf"].metadata

    response = shelf_assign_model.invoke([
        SystemMessage(
            content="Given the metadata of a particular shelf and file, your task is to simply decide if the file belongs to the shelf or not."
        ),
        HumanMessage(
            content=f"shelf_metadata: {shelf_metadata}\nfile_metadata: {file_metadata}"
        )
    ])

    return {
        "shelf_matches": [
            ShelfMatch(
                shelf=state["shelf"],
                result=response
            )
        ]
    }

def synthesizer(state: State):
    shelf_matches = state["shelf_matches"]

    highest_confidence = 0
    assigned_shelf = None

    for match in shelf_matches:
        if match.result.belongs_to_shelf and match.result.confidence >= highest_confidence:
            highest_confidence = match.result.confidence
            assigned_shelf = match.shelf

    return {
        "assigned_shelf": assigned_shelf
    }

def place_file_into_shelf(state: State):
    """Places the given file into the assigned Shelf"""

    shelf = state["assigned_shelf"]
    file = state["file"]

    shelf.files.append(file)

    print(f"Successfully Placed {file.metadata.file_name} into {shelf.metadata.shelf_name}")

    return {}

def create_shelf(state: State):
    metadata = shelf_creator.invoke([
        SystemMessage(
            content="Given the metadata of a file, create metadata for a new shelf that would broadly contain this and similar files."
        ),
        HumanMessage(
            content=f"file_metadata: {state['file_meta_data']}"
        )
    ])

    shelf = Shelf(
        metadata=metadata,
        files=[state["file"]]
    )

    result = interrupt(value={
        "interrupt_type": "shelf_create",
        "shelf": shelf.model_dump()
    })

    edited_shelf = Shelf.model_validate(result)

    if edited_shelf is None:
        print("ERROR! UNEXPECTED RESULT!")
        sys.exit()

    shelves.append(edited_shelf)

    return {
        "assigned_shelf": edited_shelf
    }

def assign_shelf_workers(state: State):
    shelves = state["shelves"]

    if len(shelves) == 0:
        return "synthesizer"

    return [ Send("check_valid_shelf", { "shelf": shelf, "file": state["file"] }) for shelf in shelves ]

def decide_creation(state: State):
    assigned_shelf = state["assigned_shelf"]

    if assigned_shelf == None:
        return "Action_newShelf"

    return "Action_existingShelf"

graph = StateGraph(State)

graph.add_node("extract_markdown", extract_markdown)
graph.add_node("extract_metadata", extract_metadata)
graph.add_node("create_file", create_file)
graph.add_node("fetch_shelves", fetch_shelves)
graph.add_node("check_valid_shelf", check_valid_shelf)
graph.add_node("synthesizer", synthesizer)
graph.add_node("place_file_into_shelf", place_file_into_shelf)
graph.add_node("create_shelf", create_shelf)

graph.add_edge(START, "extract_markdown")
graph.add_edge("extract_markdown", "extract_metadata")
graph.add_edge("extract_metadata", "create_file")
graph.add_edge("create_file", "fetch_shelves")

graph.add_conditional_edges(
    "fetch_shelves",
    assign_shelf_workers,
    ["check_valid_shelf", "synthesizer"]
)

graph.add_edge("check_valid_shelf", "synthesizer")

graph.add_conditional_edges(
    "synthesizer",
    decide_creation,
    {
        "Action_newShelf": "create_shelf",
        "Action_existingShelf": "place_file_into_shelf"
    }
)

graph.add_edge("create_shelf", END)
graph.add_edge("place_file_into_shelf",END)

checkpointer = InMemorySaver()

agent = graph.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id":"test_12"
    }
}

interrupt_id = None
interrupt_value = None

async def main(resume_value = None):

    global interrupt_value, interrupt_id
    stream = None
    if resume_value is None:
        stream = await agent.astream_events(
            {
                "file_name": "Global_warming.pdf"
            },
            version="v3",
            config=config,
        )
    else:
        stream = await agent.astream_events(
            Command(resume=resume_value),
            version="v3",
            config=config,
        )

    # async for message in stream.messages:
    #     print(message.node, flush=True)
    #     async for token in message.text:
    #         print(token, end="", flush=True)


    async for event in stream:

        if event["method"] == "messages":
            message = event["params"]

            if "content" in message["data"][0]:
                print(message["data"][0]["content"]["text"], end="", flush=True)
            elif "delta" in message["data"][0]:
                print(message["data"][0]["delta"]["text"], end="", flush = True)

        elif event["method"] == "values":

            interrupts = event["params"]["interrupts"]

            if interrupts:
                interrupt = interrupts[0]

                interrupt_id = interrupt.id
                interrupt_value = interrupt.value

                print("VAlue is ", interrupt_value)


asyncio.run(main())

if interrupt_value["interrupt_type"] == "shelf_create":
    print("Hello, the given file does not have an existing shelf.\n Are you ok with creating a new shelf with the following structure?")
    print(interrupt_value["shelf"])

    user_input = input("Answer with either Yes/No")

    if user_input == "Yes":
        asyncio.run(main(interrupt_value["shelf"]))
    else:
        shelf_details = input("Enter the following details separated by |. shelf_name, shelf_description")
        
        shelf_name, shelf_description = shelf_details.split("|")

        interrupt_value["shelf"]["metadata"]["shelf_name"] = shelf_name 
        interrupt_value["shelf"]["metadata"]["shelf_description"] = shelf_description

        asyncio.run(main(interrupt_value["shelf"]))
    print(shelves)

#TODO: Figure out a way to not print the same LLM stream after interrupt on a node.
#TODO: Include HIL System right after shelf assignment to get their confirmation.
#TODO: Figure out retrieval pipeline.