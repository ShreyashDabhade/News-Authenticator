from flask import Flask, render_template, request
from crewai import Crew, Process, Task, Agent, LLM
from crewai_tools import SerperDevTool
from GoogleNews import GoogleNews
from crewai.tools import tool
from GoogleNews import GoogleNews
import os

app = Flask(__name__)

# Setup for AI tools
os.environ['GEMINI_API_KEY'] = "AIzaSyCgi3Riy83jAc8sLgiQuKwx_2tLdX39uIE"  # Enter your actual Gemini API key
os.environ['SERPER_API_KEY'] = "79fccfb003963fb3345cba77d4158201226ca78f"  # Enter your actual Serper API key

llm = LLM(model="gemini/gemini-1.5-flash")

# Initialize Serper tool
serper_tool = SerperDevTool()

# Creating Googlenewsapi tool
@tool("GoogleNewsAPI")
def googlenews_tool(question: str, num_results: int = 10) -> str:
    """Searches Google News articles for the given query and returns the relevant results."""

    try:
        # Initialize GoogleNews
        googlenews = GoogleNews(lang='en', region='global')
        googlenews.search(question)
        results = googlenews.result()[:num_results]

        if not results:
            return f"No results found for query: {question}."

        # Collect articles and format them
        articles = []
        for item in results:
            articles.append({
                "title": item.get('title', 'No Title'),
                "link": item.get('link', 'No Link'),
                "source": item.get('source', 'Unknown Source'),
                "published_date": item.get('date', 'Unknown Date'),
            })

        # Return formatted articles as a string
        formatted_results = "\n".join([
            f"Title: {article['title']}\nSource: {article['source']}\nPublished: {article['published_date']}\nLink: {article['link']}\n"
            for article in articles
        ])

        return formatted_results

    except Exception as e:
        return f"Error while fetching news articles: {str(e)}"


# Agents Setup

## creating a researcher agent
news_researcher=Agent(
  role="Data Collection Agent",
  goal="Gather reliable and relevant data about the {input} news from the web "
         "using both the Serper tool and the GoogleNewsAPI tool. Provide a "
         "comprehensive, structured summary with proper references and source links.",
  backstory=(
        "You're a hyper-focused research assistant code-named 'Infoseek,' designed "
        "to sift through online information to collect similar and relevant news articles "
        "for verification. You utilize two powerful tools: the Serper tool for general web "
        "scraping and search engine results, and the GoogleNewsAPI tool for fetching "
        "the latest news articles directly from reputable sources. "
        "\n\n"
        "Your responsibility is to collect, organize, and present raw, unbiased data "
        "from these two tools in a clear, concise format, ensuring all gathered information "
        "is credible and relevant. You do not analyze or verify the data but focus on delivering "
        "a detailed and well-organized summary with proper references and clickable source links. "
        "This data will serve as the foundation for the next agent in the pipeline to assess the "
        "truthfulness of the news."
  ),
  llm=llm,
  allow_delegation=True
  )

## creating a verifier agent
news_verifier = Agent(
  role='Verification Agent',
  goal=   "Verify the authenticity of the {input} news by analyzing it "
          "against the data provided by the Data Collection Agent. "
          "Deliver a clear verdict on whether the news is genuine or fake, "
          "with references supporting the conclusion.",

  backstory=(
                "You're an impartial fact-checker code-named 'VeriFact,' "
                "designed to analyze and cross-verify news for accuracy. "
                "You rely exclusively on the {input} news and the evidence "
                "provided by the Data Collection Agent to authenticate the story. "
                "Your role is to produce a transparent, evidence-backed report "
                "that includes a clear verdict, reasoning, and references. "
                "You uphold a strict code of neutrality, focusing on facts and "
                "avoiding personal bias or unverifiable claims."
  ),
  llm=llm,
  allow_delegation=False
  )

## creating a news presenter agent
news_presenter = Agent(
  role='Presentation Agent',
  goal= "Deliver the final verdict on the {input} news in an aesthetically pleasing "
        "and user-friendly format based on the output provided by the Verification Agent.",

  backstory= (
              "You're a presentation expert tasked with crafting a polished and visually "
              "appealing output for the user. You rely solely on the verdict, reasoning, "
              "and references provided by the Verification Agent. Your role is to ensure "
              "the information is presented clearly, accurately, and engagingly without introducing new content or personal opinions."
              "Your primary objective is to enhance user understanding through effective design and structure."
  ),
  llm=llm,
  allow_delegation=False,
)

# Creating tasks

## Researcher Task
news_researcher_task = Task(
     description=(
        "Gather relevant and credible news articles related to the {input} news "
        "by prioritizing the use of the Serper tool to scrape the web. Focus on retrieving data from this tool first and use it thoroughly to find the latest data to stay updated. "
        "Also, use the Google News tool to search for similar news articles; if found, confirm their existence and then use serper_tool to search for that exact news article. "
        "Do not simply feed the {input} news as it is; instead, use more general keywords related to the event to broaden the search terms. "
        "Articles should come from trustworthy sources that provide additional context or details similar to the {input} news. Summarize the key points and provide a clear list "
        "of sources and links for each article found. The goal is to present factual, unbiased data that can be cross-verified by the verification agent. "
        "\n\nAdditional Instructions: In every response, ensure that the analysis remains strictly unbiased, presenting all viewpoints fairly. "
        "Include relevant background context (e.g., historical background, related events, key statistics, or trends) to help the user fully understand the topic. "
        "Clearly cite all sources and note any uncertainties or areas needing further verification."
    ),
    expected_output = 'A structured summary of relevant news articles with source links and brief descriptions.',
    tools=[serper_tool, googlenews_tool],
    agent = news_researcher,
     run = lambda inputs, tools: tools[0].run(search_query=inputs['input'])
)


## verifier Task
news_verification_task = Task(
    description=(
        "Verify the authenticity of the {input} news by comparing it to the data provided by the Data Collection Agent. "
        "Use your analysis to assess whether "
        "the news is genuine or fake. Provide a clear verdict and support your conclusion "
        "with specific references to the collected sources. Your goal is to use the provided evidence from Serper "
        "to assess whether the news is genuine or fake. Provide a clear verdict based "
        "exclusively on this data, with reasoning and references."
        "Do not add any of your biased verdict, only use the data provided by news_researcher to come to final verdict."
        "\n\nAdditional Instructions: Ensure that your analysis remains strictly unbiased and includes any relevant contextual"
        "background that may help the user understand the broader implications of the news. "
        "Clearly cite all sources and indicate any uncertainties or areas requiring further verification."
    ),
    expected_output = 'A detailed verdict report on the authenticity of the news, with supporting references.',
    agent = news_verifier
)

## Presenter Task
news_presentation_task = Task(
     description=(
        "Based on the output from the Verification Agent, present the final news verdict in a consistent, "
        "user-friendly format. Highlight the verdict prominently and include structured sections for reasoning "
        "and references. Ensure references are clickable and the entire presentation maintains clarity and readability." 
        "Present the references in an anchor tag so that user can directly click on it and go to the reference site." 
        "One precaution you have to take while creating anchor tag of the site is that the links should be only till .html remove the &ved= and the contain after it."
        "FINAL VERDICT should be <h1> heading tag and the Verdict should be <h3> heading tag"
        "\n\nAdditional Instructions: Present the information in an unbiased manner, incorporating additional context and background details"
        "(such as historical context, related events, or key statistics) to help the user gain a comprehensive understanding of the topic. "
        "Clearly cite all sources and note any uncertainties."
    ),
    expected_output=(
            " FINAL VERDICT \n"
            "🎯 Verdict: \n"
            "==================== REASONING ===================="
            "'reasoning'\n"
            "==================== REFERENCES ===================="
            "'references'\n"
            "==================================================="
        "Ensure all references are accurate and clickable, and do not introduce new content or analysis."
    ),
    agent = news_presenter
)

crew = Crew(
    agents=[news_researcher, news_verifier, news_presenter],
    tasks=[news_researcher_task, news_verification_task, news_presentation_task],
    process=Process.sequential,
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify_news():
    input_news = request.form.get("news")
    if not input_news:
        return render_template("index.html", error="Please enter some news to verify.")

    try:
        result = crew.kickoff(inputs={"input": input_news})

        result_str = str(result)
        formatted_result = result_str.replace('\n', '<br>')
        formatted_result = formatted_result.replace('Verdict:', '<strong style="color: #1a73e8">Verdict:</strong>')
        formatted_result = formatted_result.replace('Reasoning:', '<strong style="color: #1a73e8">Reasoning:</strong>')
        formatted_result = formatted_result.replace('Sources:', '<strong style="color: #1a73e8">Sources:</strong>')
        
        return render_template("result.html", verdict=formatted_result)
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        return render_template("index.html", error=error_msg)

if __name__ == "__main__":
    app.run(debug=True)
