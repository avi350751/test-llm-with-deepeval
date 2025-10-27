from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from dotenv import load_dotenv

load_dotenv()

cpm = ContextualPrecisionMetric()
test_case = LLMTestCase(
    input = "What is the capital of Argentina?",
    actual_output="Buenos Aires",
    retrieval_context=["Argentina's capital city is Buenos Aires, famous for its European-style architecture and rich cultural life."],
    expected_output="Buenos Aires"
)

cpm.measure(test_case)
print(f"Contextual Precision Score: {cpm.score}")
print(f"CPM success: {cpm.success}")