from verl.utils.reward_score.entropy_math import compute_score


def entropy_math_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Entropy-style mathematical scoring function for individual samples.

    This function uses the integrated entropy_math module which handles both
    \\boxed{} and Answer: formats, provides better mathematical equivalence
    checking, and gives consistent output format across all datasets.

    Args:
        data_source (str): The source dataset identifier (ignored in entropy approach)
        solution_str (str): The solution string to be evaluated
        ground_truth (str): The ground truth answer for comparison
        extra_info (dict, optional): Additional information (unused)
        sandbox_fusion_url (str, optional): Sandbox URL (unused)
        concurrent_semaphore: Semaphore for concurrency control (unused)
        memory_limit_mb (int, optional): Memory limit (unused)

    Returns:
        float or dict: The computed score. Returns a dictionary with detailed
                      information if available, otherwise a float score.
    """

    try:
        result = compute_score(solution_str, str(ground_truth))

        # Normalize output format to be consistent with VERL expectations
        if isinstance(result, dict):
            return result
        elif isinstance(result, (int, float, bool)):
            return float(result)
        else:
            return float(result[0])

    except Exception as e:
        print(f"Error computing entropy math score: {e}")
        return 0.0
