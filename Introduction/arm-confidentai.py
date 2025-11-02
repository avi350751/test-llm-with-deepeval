from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from dotenv import load_dotenv

load_dotenv()

arm = AnswerRelevancyMetric()
test_case = LLMTestCase(
    input = "What is the capital of France?",
    actual_output="Paris",
    retrieval_context=["France's capital city is Paris, known for its art, fashion, gastronomy, and culture."]
)

evaluate(test_cases=[test_case], metrics=[arm])