from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------- Input schemas ----------

class PestIdentifierInput(BaseModel):
    crop: str = Field(description="Name of the crop (e.g. wheat, rice, tomato)")
    symptoms: str = Field(description="Visible symptoms on the crop (e.g. yellowing leaves, holes, white patches)")


class SchemeFinderInput(BaseModel):
    state: str = Field(description="Indian state of the farmer (e.g. Punjab, Maharashtra)")
    crop: str = Field(description="Crop being cultivated")
    category: str = Field(description="Farmer land-holding category: small / marginal / large")


# ---------- Tool factory functions ----------

def build_pest_identifier(llm: ChatGroq) -> StructuredTool:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an agricultural expert. Respond only in the structured format requested. "
            "Be concise and practical.",
        ),
        (
            "human",
            """Crop: {crop}
Observed symptoms: {symptoms}

Respond strictly in this format:
PEST: <most likely pest name>
EXPLANATION: <2-3 sentence explanation of why this pest matches the symptoms>
TREATMENT: <specific, actionable treatment steps>""",
        ),
    ])

    def identify_pest(crop: str, symptoms: str) -> str:
        chain = prompt | llm
        return chain.invoke({"crop": crop, "symptoms": symptoms}).content

    return StructuredTool.from_function(
        func=identify_pest,
        name="pest_identifier",
        description=(
            "Identifies the most likely pest or disease affecting a crop based on the crop name "
            "and symptoms. Returns pest name, explanation, and treatment. "
            "Use this when a farmer describes physical problems with their crop."
        ),
        args_schema=PestIdentifierInput,
    )


def build_scheme_finder(llm: ChatGroq) -> StructuredTool:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert on Indian central and state government agricultural schemes. "
            "Include both central and state-level schemes where applicable.",
        ),
        (
            "human",
            """Farmer profile:
- State: {state}
- Crop: {crop}
- Category: {category}

List 4-5 relevant government schemes. For each use this format:
SCHEME: <official scheme name>
DESCRIPTION: <what the scheme offers in 1-2 sentences>
ELIGIBILITY: <key eligibility conditions>
---""",
        ),
    ])

    def find_schemes(state: str, crop: str, category: str) -> str:
        chain = prompt | llm
        return chain.invoke({"state": state, "crop": crop, "category": category}).content

    return StructuredTool.from_function(
        func=find_schemes,
        name="scheme_finder",
        description=(
            "Finds relevant Indian government agricultural schemes for a farmer based on their "
            "state, crop, and land-holding category. Returns scheme names, descriptions, and "
            "eligibility. Use this when a farmer asks about subsidies, schemes, or financial help."
        ),
        args_schema=SchemeFinderInput,
    )
