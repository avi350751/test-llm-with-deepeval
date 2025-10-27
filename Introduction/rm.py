from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from dotenv import load_dotenv
load_dotenv()

rm = ContextualRecallMetric()
test_case = LLMTestCase(
    input = "Which are the four famous authors of the late Victorian era?",
    actual_output="Thomas Hardy, George Elliot, OScar Wilde and H. G Wells",
    expected_output="Hardy, Mary Ann Evans (George Eliot), Oscar Wilde, and H. G. Wells",
    retrieval_context=["Books written during the later Victorian era explored similar themes of social class and romance."
    "Notable authors from this period include Thomas Hardy, George Eliot, Oscar Wilde, Jane, Austen, Liz Braddon, H. G. Wells."]
)

rm.measure(test_case)
print(f"Contextual Relevance score: {rm.score}")
print(f"CRM success: {rm.success}")