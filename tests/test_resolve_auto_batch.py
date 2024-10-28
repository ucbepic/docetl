
import pytest
from docetl.operations.resolve import ResolveOperation
from tests.conftest import api_wrapper
@pytest.fixture(scope='session')
def resolve_config():
    return {
        "name": "name_email_resolver",
        "type": "resolve",
        "blocking_keys": ["name"],
        "blocking_threshold": 0,
        "comparison_prompt": "Compare the following two entries and determine if they likely refer to the same person: Person 1: {{ input1 }} Person 2: {{ input2 }} Return true if they likely match, false otherwise.",
        "output": {"schema": {"name": "string", "email": "string"}},
        "embedding_model": "text-embedding-3-small",
        "comparison_model": "gpt-4o-mini",
        "resolution_model": "gpt-4o-mini",
        "resolution_prompt": "Given the following list of similar entries, determine one common name and email. {{ inputs }}",
    }


@pytest.fixture(scope='session')
def resolve_sample_data():
    return [
        {"name": "John Doe", "email": "john@example.com"},
        {"name": "John D.", "email": "johnd@example.com"},
        {"name": "J. Smith", "email": "jane@example.com"},
        {"name": "J. Smith", "email": "jsmith@example.com"},
        {"name": "Emily Brown", "email": "emilyb@example.com"},
        {"name": "Emily B.", "email": "brown.em@example.com"},
        {"name": "Chris White", "email": "chris@example.com"},
        {"name": "C. White", "email": "cwhite@example.com"},
        {"name": "R. Davis", "email": "rdavis@example.com"},
        {"name": "Laura Martinez", "email": "laura.m@example.com"},
        {"name": "L. Martinez", "email": "lmartinez@example.com"},
        {"name": "William Clark", "email": "willclark@example.com"},
        {"name": "W. Clark", "email": "wclark@example.com"},
        {"name": "Megan Allen", "email": "megan.allen@example.com"},
        {"name": "M. Allen", "email": "mallen@example.com"},
        {"name": "Brian Wilson", "email": "brianw@example.com"},
        {"name": "B. Wilson", "email": "bwilson@example.com"},
        {"name": "Nancy Young", "email": "nancy.young@example.com"},
        {"name": "N. Young", "email": "nyoung@example.com"},
        {"name": "Paul Walker", "email": "paulw@example.com"},
        {"name": "P. Walker", "email": "pwalker@example.com"},
        {"name": "Kimberly Adams", "email": "kim.adams@example.com"},
        {"name": "K. Adams", "email": "kadams@example.com"},
        {"name": "Joshua Green", "email": "josh.green@example.com"},
        {"name": "J. Green", "email": "jgreen@example.com"},
        {"name": "Ethan Phillips", "email": "ethanp@example.com"},
        {"name": "E. Phillips", "email": "ephillips@example.com"},
        {"name": "Victoria King", "email": "victoria.k@example.com"},
        {"name": "V. King", "email": "vking@example.com"},
        {"name": "Olivia Turner", "email": "olivia.turner@example.com"},
        {"name": "O. Turner", "email": "oturner@example.com"},
        {"name": "Daniel Moore", "email": "danmoore@example.com"},
        {"name": "D. Moore", "email": "dmoore@example.com"},
        {"name": "Grace Evans", "email": "grace.e@example.com"},
        {"name": "G. Evans", "email": "gevans@example.com"},
        {"name": "Kevin Hall", "email": "khall@example.com"},
        {"name": "K. Hall", "email": "kevin.h@example.com"},
        {"name": "Michael Black", "email": "m.black@example.com"},
        {"name": "M. Black", "email": "mblack@example.com"},
        {"name": "Sarah White", "email": "sarahw@example.com"},
        {"name": "S. White", "email": "swhite@example.com"},
        {"name": "Samuel Hill", "email": "samuel.h@example.com"},
        {"name": "Sam Hill", "email": "shill@example.com"},
        {"name": "Jessica Green", "email": "jess.green@example.com"},
        {"name": "J. Green", "email": "jgreen1@example.com"},
        {"name": "Alexander Young", "email": "alex.young@example.com"},
        {"name": "A. Young", "email": "ayoung@example.com"},
        {"name": "Rachel Brown", "email": "rachel.brown@example.com"},
        {"name": "R. Brown", "email": "rbrown@example.com"},
        {"name": "Jonathan Adams", "email": "jon.adams@example.com"},
        {"name": "J. Adams", "email": "jadams@example.com"},
        {"name": "Olivia Harris", "email": "oliviah@example.com"},
        {"name": "O. Harris", "email": "oharris@example.com"},
        {"name": "Benjamin Parker", "email": "ben.parker@example.com"},
        {"name": "B. Parker", "email": "bparker@example.com"},
        {"name": "Christina Lewis", "email": "christina.lewis@example.com"},
        {"name": "C. Lewis", "email": "clewis@example.com"},
        {"name": "Ethan Baker", "email": "ethanb@example.com"},
        {"name": "E. Baker", "email": "ebaker@example.com"},
        {"name": "Abigail Carter", "email": "abigail.c@example.com"},
        {"name": "A. Carter", "email": "acarter@example.com"},
        {"name": "C. Scott", "email": "cscott@example.com"},
        {"name": "Sophia King", "email": "sophiak@example.com"},
        {"name": "S. King", "email": "s.king@example.com"},
        {"name": "Jacob Wright", "email": "jacob.wright@example.com"},
        {"name": "J. Wright", "email": "jwright@example.com"},
        {"name": "Isabella Mitchell", "email": "isabella.m@example.com"},
        {"name": "I. Mitchell", "email": "imitchell@example.com"},
        {"name": "Daniel Martinez", "email": "dan.m@example.com"},
        {"name": "D. Martinez", "email": "dmartinez@example.com"},
        {"name": "Emma Perez", "email": "emmap@example.com"},
        {"name": "E. Perez", "email": "eperez@example.com"},
        {"name": "Christopher Nelson", "email": "chris.n@example.com"},
        {"name": "C. Nelson", "email": "cnelson@example.com"},
        {"name": "Grace Carter", "email": "gcarter@example.com"},
        {"name": "G. Carter", "email": "grace.c@example.com"},
        {"name": "Lucas Davis", "email": "lucasd@example.com"},
        {"name": "L. Davis", "email": "ldavis@example.com"},
        {"name": "Mia Robinson", "email": "mia.r@example.com"},
        {"name": "M. Robinson", "email": "mrobinson@example.com"},
        {"name": "Henry Garcia", "email": "henry.g@example.com"},
        {"name": "H. Garcia", "email": "hgarcia@example.com"},
        {"name": "Amelia Rodriguez", "email": "amelia.r@example.com"},
        {"name": "A. Rodriguez", "email": "arodriguez@example.com"},
        {"name": "David Walker", "email": "david.w@example.com"},
        {"name": "D. Walker", "email": "dwalker@example.com"}

    ]


def test_resolve_operation(
    resolve_config, max_threads, resolve_sample_data, api_wrapper
):
    operation = ResolveOperation(
        api_wrapper, resolve_config, "text-embedding-3-small", max_threads
    )
    results, cost = operation.execute(resolve_sample_data[:25])

    distinct_names = set(result["name"] for result in results)
    print(distinct_names)
    assert len(distinct_names) < len(results)
