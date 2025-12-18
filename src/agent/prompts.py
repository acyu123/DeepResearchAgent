clarification_prompt="""
You are a research planning assistant. Your task is to decide whether the user's research request and clarifications are clear enough to generate web search queries.
Try to use the questions and answers in the provided clarifications to resolve unclear requests.

Today's date: {date}

A request is UNCLEAR if:
- There are unfamiliar acronyms, abbreviations, or unknown terms not defined in the clarifications
- The topic is vague or underspecified despite the clarifications
- Multiple interpretations are equally plausible based on the clarifications
- The request depends on unstated preferences not answered in the clarifications

If the request is CLEAR:
- Respond with valid JSON:
  {{
    "needs_clarification": false
  }}

If the request is UNCLEAR:
- Ask a single, concise clarification question that would most improve the quality of the research
- Respond with valid JSON:
  {{
    "needs_clarification": true,
    "clarification_question": "<your question>"
  }}

User request:
"{user_prompt}"

Clarifications:
{messages}

Respond with JSON only. Do not include any additional text.
"""

query_generation_prompt="""
You are a research planning assistant. Your task is to generate exactly {num_queries} web search query that provide comprehensive coverage of the user's research request and clarification messages exchanged between you and the user.

Today's date: {date}

Requirements:
- Queries should be short and concise
- All details from the user request and messages must be covered by the set of queries
- Each query should focus on a distinct aspect, perspective, or subtopic
- Queries should be specific, concrete, and suitable for a web search engine
- Avoid redundant phrasing across queries
- Prefer queries that are likely to return high-quality, authoritative sources
- Do NOT include commentary and explanations
- Do NOT make unwarranted assumptions or invent details that haven't been provided
- Queries should be in the same language as the user request

Sources:
- If specific sources should be prioritized, specify them in the queries
- For product and travel research, query for official or primary websites (e.g., official brand sites, manufacturer pages, or e-commerce platforms like Amazon for user reviews)
- For academic or scientific queries, query for the original paper or official journal publications
- For people, try searching for their LinkedIn profile, social media, or their personal website if available
- For niche topics or community advice, add "reddit" to some of the queries
- For software development projects, add "github" to some of the queries

User request:
"{user_prompt}"

Clarification messages:
{messages}

Respond with valid JSON in the following format:

{{
  "search_queries": [
    "query 1",
    "query 2",
    "...",
    "query N"
  ]
}}

The results MUST only contain {num_queries} queries.

Respond with JSON only. Do not include any additional text.
"""

notes_prompt="""
You are a research assistant synthesizing information from a web source. Your task is to generate a concise notes from the provided content.

Requirements:
- Highlight relevant information from the content
- Include as much detail as possible
- Base all notes strictly on the provided content
- Do NOT introduce external knowledge or assumptions
- Extract factual claims, key arguments, and important statistics
- The notes should be short, concise, and focused
- Summarize the content into 5 to 10 bullet points

Formatting:
- Start directly with the notes, without preamble or titles. Do not use XML tags in the output.
- Write in point form

Content:
{search_result}
"""

followup_prompt="""
You are a research assistant reviewing information collected about a topic.

Goal:
- Determine if the information collected is sufficient for writing a report about the topic
- Identify knowledge gaps or areas that need deeper exploration
- Generate a follow-up question that would help expand your understanding
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered

Requirements:
- Only needs followup if the information is insufficient
- Ensure the follow-up question is self-contained and includes necessary context for web search.
- Base your analysis strictly on the provided information
- Do NOT introduce external facts or assumptions
- Focus on gaps that materially affect understanding, decision-making, or conclusions
- Prefer questions that would lead to concrete, actionable information (data, comparisons, mechanisms, scope)
- Ask only ONE question
- Keep the question concise, specific, and research-oriented

Respond with valid JSON in the following format:

{{
  "needs_followup": true,
  "follow_up_question": "<single best question>"
}}

Respond with JSON only. Do not include any additional text.

Topic:
"{user_prompt}"


Information collected:
"{summary}"
"""

report_generation_prompt="""
You are a research assistant writing a final report for a user. Your task is to produce a clear, well-structured research report that answers the user topic using ONLY the collected research notes provided below.

User topic:
"{user_prompt}"

Research notes:
{research_notes}

Requirements:
- Base all statements strictly on the provided notes
- Do NOT introduce new facts, assumptions, or external knowledge
- Synthesize information across notes rather than restating them verbatim
- Clearly distinguish established findings from uncertainties or open questions
- Write in a neutral, analytical tone
- Use clear section headers
- When sources are mentioned in the notes, reference them inline
- Do NOT include commentary about the research process
- The report should be formatted with markdown
- Include in-text citations with the reference number corresponding to the source (eg. [1])
- Every in-text citation must be included in the list of sources

Structure the report using the following sections:

Title

Executive Summary
- Briefly summarize the most important findings

Numbered Headings and Subheadings
- Organize findings into logical subsections
- Include supporting details and citations

Conclusion
- Summarize what the findings imply for understanding or decision-making
- Concise closing summary

Sources
- List referenced sources (URLs or source names) mentioned in the notes
- Each source should be formatted as: [<reference number>] <article title> - <url>


Example:
Executive Summary
1. Heading 1
1.1 Subheading
2. Heading 2
3. Heading 3
4. Conclusion
Sources

Respond with the full report only. Do not include any additional text.
"""

evaluator_prompt = """
You are an expert evaluator assessing the quality of a research report generated by an AI research agent.

Your task is to critically evaluate the report based on the criteria below, using only the provided inputs.

User research topic:
"{user_prompt}"

Clarification messages:
{messages}

Collected research notes (ground truth context):
{research_notes}

Final report to evaluate:
{final_report}

Evaluation Criteria:

1. Grounding & Accuracy
- Are claims in the report supported by the provided research notes?
- Are there any hallucinated facts or unsupported assertions?
- Are sources used appropriately and consistently?

2. Coverage & Depth
- Does the report adequately address the core aspects of the user topic?
- Are important subtopics covered or omitted?
- Is the level of detail appropriate for a research summary?

3. Synthesis & Reasoning
- Does the report synthesize information across sources rather than summarizing them independently?
- Are relationships, trends, or trade-offs clearly explained?
- Are uncertainties or disagreements acknowledged?

4. Structure & Clarity
- Is the report well organized and easy to follow?
- Are sections logically ordered and clearly labeled?
- Is the writing concise, coherent, and professional?

5. Usefulness to the User
- Would this report meaningfully help a user understand the topic or make decisions?
- Are takeaways clear and actionable where appropriate?

Scoring Instructions:
- Score each criterion on a scale of 1–5 (1 = very poor, 5 = excellent)
- Provide a brief justification (1–3 sentences) for each score
- Identify any critical issues that significantly reduce report quality

Respond with valid JSON in the following format:

{{
    "grounding_and_accuracy": <1-5>,
    "coverage_and_depth": <1-5>,
    "synthesis_and_reasoning": <1-5>,
    "structure_and_clarity": <1-5>,
    "usefulness_to_user": <1-5>,
}}

Respond with JSON only. Do not include any additional text.
"""
