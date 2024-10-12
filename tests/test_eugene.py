import pytest
from docetl.operations.map import MapOperation
from docetl.operations.unnest import UnnestOperation
from docetl.operations.resolve import ResolveOperation
from docetl.operations.reduce import ReduceOperation
from tests.conftest import api_wrapper


@pytest.fixture
def default_model():
    return "gpt-4o-mini"


@pytest.fixture
def max_threads():
    return 64


@pytest.fixture
def synthetic_data():
    return [
        {
            "survey_response": "the database normalization stuff was pretty hard but super interesting. breaking down tables into their most efficient form was confusing at first, but as we did more examples, i started to see how cool well-normalized schemas are. the SQL queries were really fun to learn, especially when we got into the complicated joins and subqueries. i liked how the teacher gave us real-world examples of where we'd use this stuff, it helped me understand better.",
            "class_id": "DB101",
        },
        {
            "survey_response": "erd concepts were confusing af at first, especially with many-to-many relationships and weak entities. but as we went thru the course and did more complex stuff, i got way better at making and reading these diagrams. the hands-on assignments were awesome, we got to use these concepts for real-world database design problems. the group project where we had to design a whole database system for a fake online store was my favorite part.",
            "class_id": "DB101",
        },
        {
            "survey_response": "i struggled with complex joins at first, especially outer joins and self-joins. the syntax was tricky and figuring out when to use each join took a while. but the indexing lecture blew my mind. learning about b-tree and hash indexes, and how they can make queries way faster, was so cool. i wish we had more hands-on stuff in the future, maybe with bigger datasets where we can really see how indexing and query optimization make a difference.",
            "class_id": "DB101",
        },
        {
            "survey_response": "this course covered so much stuff, from basic table design all the way to advanced query optimization. i really loved the part about transaction management. understanding ACID properties and how databases keep data safe even when lots of things are happening at once was fascinating. the demos of deadlock scenarios and how to fix them were super helpful. i also liked learning about different isolation levels and their trade-offs. this course gave me a solid foundation in database systems.",
            "class_id": "DB201",
        },
        {
            "survey_response": "learning about NoSQL databases was mind-blowing. i mostly knew about relational databases before, so it was cool to learn about other data models like document-based, key-value, and graph databases. the discussions on CAP theorem and how it affects distributed database systems were really interesting. i get that there's only so much time, but i wish we could've gone deeper into different database types. maybe there could be another course just on NoSQL and NewSQL systems for people who want to specialize in that?",
            "class_id": "DB201",
        },
        {
            "survey_response": "the group project where we had to design a database for a real-world scenario was def the best part of the course. it was hard but super rewarding, cuz we had to use pretty much everything we learned all semester. from figuring out what was needed and making the initial models to actually designing the database and making queries faster, the project tied everything together. working in a team was great practice for future jobs too. the feedback sessions with the teacher really helped us improve our designs.",
            "class_id": "DB201",
        },
        {
            "survey_response": "the lectures on database security and access control were really relevant and interesting, especially with all the data breaches and privacy stuff in the news. we learned a lot about authentication, role-based access control, and encryption. i loved the hands-on lab where we set up row-level security in PostgreSQL. since data protection is so important nowadays, i think we could even spend more time on security stuff. maybe we could look at real data breaches and talk about how good database security could've stopped them?",
            "class_id": "DB301",
        },
        {
            "survey_response": "the part about making databases faster was super useful and i've already used it in my internship. learning about query execution plans, collecting stats, and tuning indexes gave me some great tools to speed up databases. the exercises where we fixed slow queries were really eye-opening. i've used stuff like covering indexes and rewriting queries to make some database stuff way faster at my internship. the lecture on partitioning and how it affects query speed was awesome too. it'd be cool to see more real-world examples of speeding up databases in future classes.",
            "class_id": "DB301",
        },
        {
            "survey_response": "i liked learning about ACID properties and stuff, but i felt like there was a gap between the theory and actually using it. understanding Atomicity, Consistency, Isolation, and Durability was important, but i wanted to see more real examples of how these work in actual database systems. like, maybe showing us how a database makes sure a transaction is atomic, or what different isolation levels do to performance. it might be cool to have a project where we make our own simple transaction system to really get how these properties work.",
            "class_id": "DB301",
        },
        {
            "survey_response": "the guest lecture from the industry person about cloud databases was so cool and showed me where databases are going. learning about the problems and solutions in distributed database systems, like dealing with eventual consistency, sharding, and keeping things available, was mind-blowing. it was interesting to see how old database ideas are being used in the cloud. i liked hearing real examples of how big tech companies use cloud database tech. this lecture made me want to learn more about this stuff, and now i'm thinking about specializing in cloud data management for my career. it'd be awesome if we could learn more about new database tech and how it's used in the real world.",
            "class_id": "DB401",
        },
    ]


@pytest.fixture
def extract_themes_config():
    return {
        "name": "extract_themes",
        "type": "map",
        "prompt": """
        I'm teaching a class on databases. Analyze the following student survey response:

        {{ input.survey_response }}

        Extract 2-3 main themes from this response, each being 1-2 words. Return the themes as a list of strings.
        """,
        "output": {"schema": {"theme": "list[str]"}},
        "validate": ["len(output['theme']) >= 2"],
        "num_retries_on_validate_failure": 3,
    }


@pytest.fixture
def unnest_themes_config():
    return {"type": "unnest", "unnest_key": "theme", "name": "unnest_themes"}


@pytest.fixture
def resolve_themes_config():
    return {
        "name": "resolve_themes",
        "type": "resolve",
        "embedding_model": "text-embedding-3-small",
        "blocking_threshold": 0.7,
        "blocking_keys": ["theme"],
        "limit_comparisons": 1000,
        "comparison_prompt": """
        Compare the following two themes extracted from student survey responses about a database class:

        Theme 1: {{ input1.theme }}
        Theme 2: {{ input2.theme }}

        Are these themes similar/should they be merged?
        """,
        "resolution_prompt": """
        You are merging similar themes from student survey responses about a database class. Here are the themes to merge:

        {% for theme in inputs %}
        Theme {{ loop.index }}: {{ theme.theme }}
        {% endfor %}

        Create a single, concise theme that captures the essence of all these themes.
        """,
        "output": {"schema": {"theme": "str"}},
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def summarize_themes_config():
    return {
        "name": "summarize_themes",
        "type": "reduce",
        "reduce_key": "theme",
        "prompt": """
        I am teaching a class on databases. You are helping me analyze student survey responses. Summarize the responses for the theme: {{ inputs[0].theme }}

        Responses:
        {% for item in inputs %}
        Survey {{ loop.index }}:
        - {{ item.survey_response }}
        {% endfor %}

        Summarize the main points from the surveys expressed about this theme. Do not mention any names of students or any other identifying information.
        """,
        "output": {"schema": {"summary": "str"}},
    }


def test_database_survey_pipeline(
    synthetic_data,
    extract_themes_config,
    unnest_themes_config,
    resolve_themes_config,
    summarize_themes_config,
    default_model,
    max_threads,
    api_wrapper,
):
    # Extract themes
    extract_op = MapOperation(
        api_wrapper, extract_themes_config, default_model, max_threads
    )
    extracted_results, extract_cost = extract_op.execute(synthetic_data)

    assert len(extracted_results) == len(synthetic_data)
    assert all("theme" in result for result in extracted_results)
    assert all(len(result["theme"]) >= 2 for result in extracted_results)

    # Unnest themes
    unnest_op = UnnestOperation(
        api_wrapper, unnest_themes_config, default_model, max_threads
    )
    unnested_results, unnest_cost = unnest_op.execute(extracted_results)

    assert len(unnested_results) > len(extracted_results)
    assert all("theme" in result for result in unnested_results)

    # Resolve themes
    resolve_op = ResolveOperation(
        api_wrapper, resolve_themes_config, default_model, max_threads
    )
    resolved_results, resolve_cost = resolve_op.execute(unnested_results)

    assert len(resolved_results) <= len(unnested_results)
    assert all("theme" in result for result in resolved_results)

    # Summarize themes
    summarize_op = ReduceOperation(
        api_wrapper, summarize_themes_config, default_model, max_threads
    )
    summarized_results, summarize_cost = summarize_op.execute(resolved_results)

    assert len(summarized_results) <= len(resolved_results)
    assert all("summary" in result for result in summarized_results)

    total_cost = extract_cost + unnest_cost + resolve_cost + summarize_cost
