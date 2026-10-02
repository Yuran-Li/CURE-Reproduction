"""Answer reward shared with the local PAG runs.

The numeric answer check is the same function PAG uses in
``verl/utils/reward_score/math_verify.py``:

* gold answer is wrapped as ``\\boxed{...}``
* ``LatexExtractionConfig(boxed_match_priority=0)``
* only the last 300 characters of the model output are verified

``math-verify`` must be the same release as the PAG environment (0.9.0).
CURE's original scorer did not truncate and did not set ``boxed_match_priority``.
"""

# Sources accepted by the upstream CURE recipe.
_CURE_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
    "OlympiadBench/OE_TO_maths_en_COMP",
    "zwhe99/DeepMath-103K",
}

# Sources on the PAG parquet files used for the controlled comparison.
_PAG_SOURCES = {
    "DigitalLearningGmbH/MATH-lighteval",  # datasets/math7500.parquet
    "HuggingFaceH4/MATH-500",  # datasets/math500.parquet
    "HuggingFaceH4/aime_2024",
    "opencompass/AIME2025",
    "math-ai/minervamath",
    "dapo",  # datasets/dapo17k.parquet
}

ALLOWED_DATA_SOURCES = _CURE_SOURCES | _PAG_SOURCES


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source not in ALLOWED_DATA_SOURCES:
        raise ValueError(f"Unsupported data source for reasoning task: {data_source}")
    return compute_math_score(solution_str, ground_truth)


def compute_math_score(model_output: str, ground_truth: str) -> dict:
    """PAG ``math_verify.compute_score``. Returns ``{"score", "acc", "pred"}``."""
    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError:
        print("To use Math-Verify, please install it first by running `pip install math-verify==0.9.0`.")
        raise

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    )
    ret_score = 0.0
    pred = ""

    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    # Same truncation as PAG: longest MATH-500 answer they measured is 159 chars.
    solution_str = model_output[-300:]
    try:
        ret_score, answers = verify_func([ground_truth_boxed], [solution_str])
        if answers and len(answers) > 1 and len(answers[1]) > 0:
            pred = answers[1][0]
    except TimeoutException:
        pass
    except Exception:
        pass

    return {
        "score": ret_score,
        "acc": ret_score,
        "pred": pred,
    }
