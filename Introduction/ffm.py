from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from dotenv import load_dotenv

load_dotenv()

fm = FaithfulnessMetric()
test_case = LLMTestCase(
    input = "Who wrote 'Pride and Prejudice'?",
    actual_output="Jane Austen",
    expected_output="Jane Austen",
    retrieval_context=["'Pride and Prejudice' is a novel written by Jane Austen, first published in 1813."]
)

fm.measure(test_case)
print(f"Faithfulness score: {fm.score}")
print(f"Faithfulness success: {fm.success}")