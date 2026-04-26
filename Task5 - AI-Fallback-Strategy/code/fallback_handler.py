# fallback_handler.py

class PrimaryProvider:
    def generate(self, prompt):
        # Simulating API timeout / failure
        raise TimeoutError("Primary provider timeout")


class FallbackProvider:
    def generate(self, prompt):
        return f"Generated successfully using fallback provider for: {prompt}"


def retry_request(provider, prompt, retries=2):
    """
    Retry primary provider before switching to fallback
    """
    for attempt in range(retries):
        try:
            return provider.generate(prompt)
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")

    return None


def queue_background_job(prompt):
    """
    If both providers fail, move task to background queue
    """
    return {
        "status": "queued",
        "message": "Request moved to background processing",
        "prompt": prompt
    }


def generate_with_fallback(prompt):
    primary = PrimaryProvider()
    fallback = FallbackProvider()

    print("Trying primary provider...")

    # Step 1: Retry primary provider
    result = retry_request(primary, prompt)

    if result:
        return {
            "status": "success",
            "provider": "Primary Provider",
            "result": result
        }

    print("Switching to fallback provider...")

    # Step 2: Use fallback provider
    try:
        fallback_result = fallback.generate(prompt)
        return {
            "status": "success",
            "provider": "Fallback Provider",
            "result": fallback_result
        }

    except Exception:
        # Step 3: If fallback also fails → queue job
        return queue_background_job(prompt)


if __name__ == "__main__":
    sample_prompt = "Generate a product demo video for skincare brand"

    output = generate_with_fallback(sample_prompt)
    print(output)