"""The examples the harness can run.

Each example is a Task: a text input, a fixed set of labels, the default Laya prompt
for it, and what the Karpathy loop is told about it. Everything else (inference,
scoring, the loop, the API) is written against a Task and knows nothing about any
particular example.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import EVAL_LANGUAGE
from app.tasks.intent.labels import SearchIntent

DEFAULT_TASK_ID = "search_intent"

QUESTION_TYPES = ("choice", "noul")


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    summary: str
    # Key of the state field the text goes under, and so the placeholder in a state template.
    input_field: str
    # What the input is called in the UI and in the loop's prompts.
    input_label: str
    item: str
    items: str
    # "choice" picks one of the labels. "noul" is Laya's yes/no question: labels are [no, yes].
    question_type: str
    labels: List[str]
    default_config: Dict[str, Any]
    # "The task is <loop_task> into exactly these labels: ... <loop_conventions>"
    loop_task: str
    loop_conventions: str
    # Laya reads at most 512 tokens of state, so longer inputs are cut before they reach it.
    max_input_chars: int = 1500
    # Only golden rows in this language are evaluated or trained on. None means all rows.
    language: Optional[str] = None
    supports_serp: bool = False
    # Where the golden data comes from, for the UI. None for data that is not downloadable.
    source: Optional[str] = None

    @property
    def placeholder(self) -> str:
        return "{%s}" % self.input_field

    # Laya keeps only the first 48 tokens of each option description, and the whole question
    # (instructions + all options) must fit its 192-token head budget, so the more labels a
    # task has, the shorter each description has to be.
    @property
    def max_criterion_words(self) -> int:
        return 35 if len(self.labels) <= 4 else 18

    @property
    def max_instruction_words(self) -> int:
        return 60 if len(self.labels) <= 4 else 40

    def public(self) -> Dict[str, Any]:
        """What the frontend needs to know about the task."""
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "input_field": self.input_field,
            "input_label": self.input_label,
            "item": self.item,
            "items": self.items,
            "question_type": self.question_type,
            "labels": list(self.labels),
            "label_descriptions": dict(self.default_config["criteria"]),
            "language": self.language,
            "supports_serp": self.supports_serp,
            "source": self.source,
        }


def _state(field: str) -> Dict[str, str]:
    return {field: "{%s}" % field}


_TASKS = [
    Task(
        id="search_intent",
        name="Search intent",
        summary="Classify a search query by what the searcher is trying to do.",
        input_field="query",
        input_label="Search keyword",
        item="query",
        items="queries",
        question_type="choice",
        labels=SearchIntent.all_labels(),
        default_config={
            "state_template": "{query}",
            "instructions": "Classify the search query intent based on what the user is trying to accomplish.",
            "criteria": {
                SearchIntent.INFORMATIONAL.value: "User seeks information, education, or knowledge (how to, what is, why, explain, tutorial, guide)",
                SearchIntent.NAVIGATIONAL.value: "User seeks to reach a specific website or resource (login, sign in, account, official site, brand name)",
                SearchIntent.COMMERCIAL.value: "User researches products/services before purchasing (best, top, review, comparison, vs, alternative)",
                SearchIntent.TRANSACTIONAL.value: "User intends to complete a transaction (buy, purchase, order, download, book, subscribe, price, deal)",
            },
        },
        loop_task="search-intent classification of search queries",
        loop_conventions=(
            "The gold labels were reviewed by hand against the standard definitions, using these conventions: "
            "questions, facts, people, news, entertainment and game content are informational; a named business, "
            "organisation, website, login page or event is navigational; researching products, services or businesses "
            "(reviews, comparisons, product categories, photos of a venue) is commercial; buying, booking, tickets, "
            "downloads and specific products sold in shops are transactional. The gold examples you are shown follow them."
        ),
        max_input_chars=500,
        language=EVAL_LANGUAGE,
        supports_serp=True,
        source="data/test_db.json (1,000 search queries, labels reviewed by hand)",
    ),
    Task(
        id="support_routing",
        name="Support ticket routing",
        summary="Route a banking customer's message to the team that handles it.",
        input_field="message",
        input_label="Customer message",
        item="message",
        items="messages",
        question_type="choice",
        labels=["card", "card_payment", "transfer", "top_up", "cash_withdrawal", "account"],
        default_config={
            "state_template": _state("message"),
            "instructions": "What is the customer's `message` about?",
            "criteria": {
                "card": "the card itself: ordering, delivery, activation, PIN, lost or not working",
                "card_payment": "a payment made with the card: declined, pending, charged twice, a fee or a refund",
                "transfer": "sending or receiving a bank transfer",
                "top_up": "adding money to the account",
                "cash_withdrawal": "taking cash out at an ATM",
                "account": "identity checks, personal details, closing the account or exchanging currencies",
            },
        },
        loop_task="routing of customer messages sent to a banking app's support",
        loop_conventions=(
            "The gold labels come from the Banking77 dataset, whose 77 fine-grained intents were grouped into these "
            "six by what the message is about: the physical or virtual card itself (card); something that happened to "
            "a payment made with the card, including refunds and unexpected charges on the statement (card_payment); "
            "bank transfers in or out (transfer); adding money to the account (top_up); cash machines and cash "
            "withdrawals (cash_withdrawal); identity verification, personal details, passcode, closing the account "
            "and currency exchange (account)."
        ),
        source="mteb/banking77 on Hugging Face, 77 intents grouped into 6",
    ),
    Task(
        id="prompt_injection",
        name="Prompt-injection guard",
        summary="Decide whether a prompt tries to override an AI assistant's instructions.",
        input_field="prompt",
        input_label="Prompt",
        item="prompt",
        items="prompts",
        question_type="noul",
        labels=["benign", "injection"],
        default_config={
            "state_template": _state("prompt"),
            "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
            "criteria": {
                "benign": "an ordinary request or question",
                "injection": "tries to override, reveal or bypass the assistant's instructions",
            },
        },
        loop_task="detection of prompt-injection attempts in prompts sent to an AI assistant",
        loop_conventions=(
            "The gold labels come from a prompt-injection dataset: injection means the prompt contains text that "
            "tries to override, ignore, reveal or bypass the assistant's instructions or to take on an unrestricted "
            "persona, even when it is wrapped inside an innocent-looking request; benign is every ordinary question, "
            "task or instruction, including long and unusual ones."
        ),
        source="xTRam1/safe-guard-prompt-injection on Hugging Face",
    ),
    Task(
        id="question_type",
        name="Question type",
        summary="Classify a question by the kind of answer it expects.",
        input_field="question",
        input_label="Question",
        item="question",
        items="questions",
        question_type="choice",
        labels=["abbreviation", "description", "entity", "human", "location", "number"],
        default_config={
            "state_template": _state("question"),
            "instructions": "What kind of answer does `question` ask for?",
            "criteria": {
                "abbreviation": "an abbreviation or what one stands for",
                "description": "a definition, explanation or reason",
                "entity": "a thing: an animal, product, colour, food, word or creative work",
                "human": "a person, group or organisation",
                "location": "a place: city, country, mountain or address",
                "number": "a number: count, date, distance, money or percentage",
            },
        },
        loop_task="classification of questions by the type of answer they expect",
        loop_conventions=(
            "The gold labels are the six coarse TREC question classes: abbreviation (an abbreviation or its "
            "expansion), description (definitions, explanations, reasons, manners), entity (things such as animals, "
            "foods, products, colours, languages, terms and creative works), human (individuals, groups and "
            "organisations), location (cities, countries, states, mountains and other places) and number (counts, "
            "dates, distances, money, percentages and other numeric values)."
        ),
        source="SetFit/TREC-QC on Hugging Face (coarse classes)",
    ),
    Task(
        id="news_topic",
        name="News topic",
        summary="Classify a news article by topic.",
        input_field="article",
        input_label="News article",
        item="article",
        items="articles",
        question_type="choice",
        labels=["world", "sports", "business", "sci_tech"],
        default_config={
            "state_template": _state("article"),
            "instructions": "What is the news `article` about?",
            "criteria": {
                "world": "international news, politics, conflict and diplomacy",
                "sports": "sport: matches, players, teams and tournaments",
                "business": "companies, markets, the economy and finance",
                "sci_tech": "science, technology, software, the internet and gadgets",
            },
        },
        loop_task="topic classification of short news articles (headline plus first sentences)",
        loop_conventions=(
            "The gold labels are the four AG News sections the article was published in: world, sports, business "
            "and sci_tech. An article about a technology company's earnings or lawsuit can sit in either business "
            "or sci_tech; the gold label is the section, not a judgement about the content."
        ),
        source="fancyzhx/ag_news on Hugging Face",
    ),
    Task(
        id="email_triage",
        name="Email triage",
        summary="Separate legitimate email from spam and phishing.",
        input_field="body",
        input_label="Email text",
        item="email",
        items="emails",
        question_type="choice",
        labels=["legitimate", "spam", "phishing"],
        default_config={
            "state_template": _state("body"),
            "instructions": "What kind of email is in `body`?",
            "criteria": {
                "legitimate": "a genuine personal or work email",
                "spam": "unsolicited advertising or bulk marketing",
                "phishing": "a scam that tries to steal money, credentials or personal data",
            },
        },
        loop_task="triage of emails",
        loop_conventions=(
            "The gold labels come from two public datasets: legitimate is genuine work and mailing-list email, spam "
            "is unsolicited advertising (pills, software, stocks, adult sites), and phishing is email that tries to "
            "trick the reader into giving money, credentials or personal data. The emails are lower-cased and "
            "tokenised, and long ones are cut off."
        ),
        source="SetFit/enron_spam and zefang-liu/phishing-email-dataset on Hugging Face",
    ),
    Task(
        id="toxicity",
        name="Toxicity moderation",
        summary="Decide whether a comment is toxic.",
        input_field="post",
        input_label="Comment",
        item="post",
        items="posts",
        question_type="noul",
        labels=["not_toxic", "toxic"],
        default_config={
            "state_template": _state("post"),
            "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
            "criteria": {
                "not_toxic": "an ordinary comment, even if critical or blunt",
                "toxic": "insults, threats, harassment, hate or obscene abuse",
            },
        },
        loop_task="toxicity moderation of comments posted on Wikipedia talk pages",
        loop_conventions=(
            "The gold labels come from human raters of the Jigsaw toxic-comment data: toxic covers insults, "
            "obscenity, threats, identity hate and harassment aimed at someone; not_toxic covers everything else, "
            "including heated disagreement and criticism of edits that stays civil."
        ),
        source="OxAISH-AL-LLM/wiki_toxic on Hugging Face (balanced sample)",
    ),
    Task(
        id="request_domain",
        name="LLM request routing",
        summary="Route a request for a language model by its domain.",
        input_field="request",
        input_label="Request",
        item="request",
        items="requests",
        question_type="choice",
        labels=["code", "math_or_logic", "writing", "factual_lookup", "data_analysis", "chitchat"],
        default_config={
            "state_template": _state("request"),
            "instructions": "What domain does `request` belong to?",
            "criteria": {
                "code": "software engineering, programming, refactoring, architecture, debugging",
                "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
                "writing": "creative writing, essays, emails, blog posts, copywriting",
                "factual_lookup": "facts, definitions, trivia, history",
                "data_analysis": "statistics, SQL, data manipulation, metrics",
                "chitchat": "casual conversation, greetings, small talk",
            },
        },
        loop_task="routing of requests sent to a language model by their domain",
        loop_conventions=(
            "The gold labels were assigned by where each request came from: programming exercises (code), arithmetic "
            "word problems (math_or_logic), story prompts (writing), general-knowledge questions (factual_lookup), "
            "questions to be answered from a database table (data_analysis) and lines of casual conversation "
            "(chitchat)."
        ),
        source="six public datasets on Hugging Face, one per label",
    ),
]

for _task in _TASKS:
    assert _task.question_type in QUESTION_TYPES
    assert _task.question_type != "noul" or len(_task.labels) == 2
    assert list(_task.default_config["criteria"]) == _task.labels

TASKS: Dict[str, Task] = {task.id: task for task in _TASKS}


def get_task(task_id: Optional[str]) -> Task:
    """The task with this id. No id means the default task, which is what data saved before tasks existed belongs to."""
    task = TASKS.get(task_id or DEFAULT_TASK_ID)
    if task is None:
        raise ValueError(f"Unknown example '{task_id}'. Choose one of: {', '.join(TASKS)}")
    return task
