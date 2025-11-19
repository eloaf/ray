# Asynchronous Inference with Ray Serve

**⏱️ Time to complete**: 30 minutes

This template demonstrates how to build scalable asynchronous inference services using Ray Serve. Learn how to handle long-running PDF processing tasks without blocking HTTP responses, using Celery task queues and Redis as a message broker.

## Overview

Traditional synchronous APIs block until processing completes, causing timeouts for long-running tasks. Ray Serve's asynchronous inference pattern decouples request lifetime from compute time by:

1. Accepting HTTP requests and immediately returning a task ID
2. Enqueuing work to background processors (Celery workers)
3. Allowing clients to poll for status and retrieve results

This example implements a **PDF processing service** that extracts text and generates summaries from PDF documents.

## Prerequisites

- Python 3.9+
- Ray 2.50.0+
- Redis (for message broker and result backend)

## Step 1: Setup Redis

Redis serves as both the message broker (task queue) and result backend.

**Install and start Redis (Google Colab compatible)**


```python
# Install and start Redis server
!sudo apt-get update -qq
!sudo apt-get install -y redis-server
!sudo service redis-server start

# Verify Redis is running
!redis-cli ping
```

**Alternative methods:**

- macOS: `brew install redis && brew services start redis`
- Docker: `docker run -d -p 6379:6379 redis:latest`
- [Official Redis Installation Guide](https://redis.io/docs/getting-started/installation/)

## Step 2: Install Dependencies


```python
!pip install -q ray[serve-async-inference]>=2.50.0 requests>=2.31.0 PyPDF2>=3.0.0 celery[redis]
```


## Step 3: Start the Ray Serve Application

First, let's run the complete example to see it in action:


```python
%%writefile server.py
"""
Ray Serve Asynchronous Inference - PDF Processing Example

This example shows how to build async inference services that handle long-running
tasks without blocking HTTP responses. Tasks are queued to Redis and processed by
background workers.
"""

import io
import logging
import time
from typing import Dict, Any

import requests
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from PyPDF2 import PdfReader

from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import (
    instantiate_adapter_from_config,
    task_consumer,
    task_handler,
)

logger = logging.getLogger("ray.serve")

TASK_PROCESSOR_CONFIG = TaskProcessorConfig(
    queue_name="pdf_processing_queue",
    adapter_config=CeleryAdapterConfig(
        broker_url="redis://127.0.0.1:6379/0",
        backend_url="redis://127.0.0.1:6379/0",
    ),
    max_retries=3,
    failed_task_queue_name="failed_pdfs",
    unprocessable_task_queue_name="invalid_pdfs",
)

# ============================================================================
# Request Model
# ============================================================================

class ProcessPDFRequest(BaseModel):
    """Request schema for PDF processing."""
    pdf_url: HttpUrl
    max_summary_paragraphs: int = 3


# ============================================================================
# Task Consumer - Background Worker
# ============================================================================

@serve.deployment(num_replicas=2, max_ongoing_requests=5, ray_actor_options={"num_cpus": 0.1})
@task_consumer(task_processor_config=TASK_PROCESSOR_CONFIG)
class PDFProcessor:
    """
    Background worker that processes PDF documents asynchronously.

    Configuration:
    - num_replicas=2: Run 2 worker instances
    - max_ongoing_requests=5: Each worker handles up to 5 concurrent tasks
    - max_retries=3: Retry failed tasks up to 3 times
    """

    @task_handler(name="process_pdf")
    def process_pdf(
        self, pdf_url: str, max_summary_paragraphs: int = 3
    ) -> Dict[str, Any]:
        """
        Download PDF, extract text, and generate summary.

        Args:
            pdf_url: URL to the PDF file
            max_summary_paragraphs: Number of paragraphs for summary (default: 3)

        Returns:
            Dictionary with extracted text, summary, and metadata
        """
        start_time = time.time()
        logger.info(f"Processing PDF: {pdf_url}")

        try:
            # Download PDF from URL
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Parse PDF content
            pdf_file = io.BytesIO(response.content)
            try:
                pdf_reader = PdfReader(pdf_file)
            except Exception as e:
                raise ValueError(f"Invalid PDF file: {str(e)}")

            if len(pdf_reader.pages) == 0:
                raise ValueError("PDF contains no pages")

            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            if not full_text.strip():
                raise ValueError("PDF contains no extractable text")

            # Generate summary (first N paragraphs)
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            summary = "\n\n".join(paragraphs[:max_summary_paragraphs])

            # Calculate metadata
            result = {
                "status": "success",
                "pdf_url": pdf_url,
                "page_count": len(pdf_reader.pages),
                "word_count": len(full_text.split()),
                "full_text": full_text,
                "summary": summary,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }

            logger.info(f"Processed PDF: {result['page_count']} pages, {result['word_count']} words")
            return result

        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to download PDF: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Failed to process PDF: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


# ============================================================================
# HTTP API - Ingress Deployment
# ============================================================================

fastapi_app = FastAPI(title="Async PDF Processing API")


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(fastapi_app)
class AsyncPDFAPI:
    """
    HTTP API for submitting and checking PDF processing tasks.

    Endpoints:
    - POST /process: Submit a PDF processing task
    - GET /status/{task_id}: Check task status and get results
    """

    def __init__(self, task_processor_config: TaskProcessorConfig, handler: DeploymentHandle):
        """Initialize the API with task adapter."""
        self.adapter = instantiate_adapter_from_config(task_processor_config)
        logger.info("AsyncPDFAPI initialized")

    @fastapi_app.post("/process")
    async def process_pdf(self, request: ProcessPDFRequest):
        """
        Submit a PDF processing task.

        Returns task_id immediately without waiting for processing to complete.
        Client should poll /status/{task_id} to check progress.
        """
        task_result = self.adapter.enqueue_task_sync(
            task_name="process_pdf",
            kwargs={
                "pdf_url": str(request.pdf_url),
                "max_summary_paragraphs": request.max_summary_paragraphs,
            },
        )

        logger.info(f"Enqueued task: {task_result}")

        return {
            "task_id": task_result.id,
            "status": task_result.status,
            "message": "PDF processing task submitted successfully",
        }

    @fastapi_app.get("/status/{task_id}")
    async def get_status(self, task_id: str):
        """
        Get task status and results.

        Status values:
        - PENDING: Task queued, waiting for worker
        - STARTED: Worker is processing the task
        - SUCCESS: Task completed successfully (result available)
        - FAILURE: Task failed (error message available)
        """
        status = self.adapter.get_task_status_sync(task_id)

        return {
            "task_id": task_id,
            "status": status.status,
            "result": status.result if status.status == "SUCCESS" else None,
            "error": str(status.result) if status.status == "FAILURE" else None,
        }


# ============================================================================
# Application Setup
# ============================================================================

def build_app():
    """Build and configure the Ray Serve application."""
    # Deploy background worker
    consumer = PDFProcessor.bind()

    # Deploy HTTP API
    api = AsyncPDFAPI.bind(TASK_PROCESSOR_CONFIG, consumer)

    return api


# Entry point for Ray Serve
app = build_app()
```


```python
from ray import serve
from server import app

serve.run(
    target=app,
    blocking=False
)
```

## Step 4: Test the Service

Below is the client.py file, which basically calls our above-created ray serve application to process the PDF, and then poll those task ids for the result.


```python
%%writefile client.py
"""
Example client for testing asynchronous PDF processing.

Demonstrates:
1. Submitting PDF processing tasks
2. Polling for task status
3. Retrieving results when complete
"""

import time
from typing import Dict, Any

import json

import requests


# ============================================================================
# AsyncPDFClient - Client for interacting with the async PDF API
# ============================================================================

class AsyncPDFClient:
    """Client for interacting with the async PDF processing API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client with the base URL of the API.

        Args:
            base_url: Base URL of the async PDF processing service
        """
        self.base_url = base_url.rstrip("/")

    def process_pdf(self, pdf_url: str, max_summary_paragraphs: int = 3) -> str:
        """
        Submit a PDF processing task to the server.

        This method returns immediately with a task_id without waiting
        for the PDF to be processed.

        Args:
            pdf_url: URL of the PDF file to process
            max_summary_paragraphs: Number of paragraphs to include in summary

        Returns:
            task_id: Unique identifier for tracking this task
        """
        response = requests.post(
            f"{self.base_url}/process",
            json={
                "pdf_url": pdf_url,
                "max_summary_paragraphs": max_summary_paragraphs,
            },
        )
        return response.json()["task_id"]

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.

        Args:
            task_id: The task identifier returned by process_pdf()

        Returns:
            Dictionary containing:
            - status: PENDING, STARTED, SUCCESS, or FAILURE
            - result: Task result (only present if status is SUCCESS)
            - error: Error message (only present if status is FAILURE)
        """
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete by polling its status.

        This method blocks until the task completes or times out.

        Args:
            task_id: The task identifier to wait for
            poll_interval: Seconds to wait between status checks
            timeout: Maximum seconds to wait before timing out

        Returns:
            Final task status dictionary with results

        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If task fails
        """
        start_time = time.time()

        while True:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

            # Get current task status
            status = self.get_task_status(task_id)
            state = status["status"]

            if state == "SUCCESS":
                return status
            elif state == "FAILURE":
                raise RuntimeError(f"Task failed: {status.get('error')}")
            elif state in ["PENDING", "STARTED"]:
                print(f"  Task status: {state}, waiting...")
                time.sleep(poll_interval)
            else:
                print(f"  Unknown status: {state}, waiting...")
                time.sleep(poll_interval)


# ============================================================================
# Main Example - Process Multiple PDFs
# ============================================================================

def main():
    """
    Run example PDF processing tasks.

    This demonstrates:
    1. Submitting multiple PDF processing tasks in parallel
    2. Waiting for all tasks to complete
    3. Retrieving and displaying results
    """
    client = AsyncPDFClient()

    print("=" * 70)
    print("Asynchronous PDF Processing Example")
    print("=" * 70)

    # Example: Process multiple PDFs in parallel
    print("\n" + "=" * 70)
    print("Step 1: Submitting PDF processing tasks")
    print("=" * 70)

    # List of PDFs to process
    pdf_urls = [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "https://arxiv.org/pdf/1706.03762.pdf",  # "Attention Is All You Need" paper
    ]

    # Submit all tasks (non-blocking, returns immediately)
    task_ids = []
    for i, url in enumerate(pdf_urls, 1):
        try:
            task_id = client.process_pdf(url)
            task_ids.append((task_id, url))
            print(f"   ✓ Task {i} submitted: {task_id}")
        except Exception as e:
            print(f"   ✗ Task {i} failed to submit: {e}")

    # Wait for all tasks to complete
    print("\n" + "=" * 70)
    print("Step 2: Waiting for tasks to complete")
    print("=" * 70)

    # Poll each task until it completes
    for i, (task_id, url) in enumerate(task_ids, 1):
        print(f"\nTask {i} ({url.split('/')[-1]}):")
        try:
            # This blocks until the task completes or times out
            result = client.wait_for_task(task_id, timeout=60.0)
            if result["result"]:
                res = result["result"]
                print(f"   ✓ Complete: {res['page_count']} pages, {res['word_count']} words")
                print(f"   ✓ Processing time: {res['processing_time_seconds']}s")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```


```python
# Run the client to test the async PDF processing service
!python client.py
```

### Understanding the Client Code

Now let's break down how the async workflow works. We'll use the client methods interactively:

```
import requests
import time

BASE_URL = "http://localhost:8000"
```



#### 1. Submit a PDF Processing Task

Submit returns immediately with a task ID, without waiting for processing:

```
pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

response = requests.post(
    f"{BASE_URL}/process",
    json={
        "pdf_url": pdf_url,
        "max_summary_paragraphs": 2
    }
)

task_data = response.json()
task_id = task_data["task_id"]["id"]

print(f"✓ Task submitted!")
print(f"  Task ID: {task_id}")
print(f"  Status: {task_data['status']}")
```

#### 2. Poll for Task Status

Check the task status to see if it's complete:

```
# Check status (may need to run this cell multiple times)
response = requests.get(f"{BASE_URL}/status/{task_id}")
status_data = response.json()

print(f"Task status: {status_data['status']}")

if status_data['status'] == 'SUCCESS':
    result = status_data['result']
    print(f"\n✓ Complete!")
    print(f"  Pages: {result['page_count']}")
    print(f"  Words: {result['word_count']}")
    print(f"  Time: {result['processing_time_seconds']}s")
elif status_data['status'] == 'FAILURE':
    print(f"\n✗ Failed: {status_data.get('error')}")
else:
    print(f"  Still processing... (Status: {status_data['status']})")
```

#### 3. Wait for Completion

Or use a polling loop to automatically wait:

```
# Submit a new task and wait for completion
pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"

response = requests.post(
    f"{BASE_URL}/process",
    json={"pdf_url": pdf_url, "max_summary_paragraphs": 3}
)

task_id = response.json()["task_id"]["id"]
print(f"Task submitted: {task_id}\n")

# Poll until complete
max_attempts = 40
for attempt in range(max_attempts):
    response = requests.get(f"{BASE_URL}/status/{task_id}")
    status_data = response.json()
    
    if status_data['status'] == 'SUCCESS':
        result = status_data['result']
        print(f"\n✓ Complete!")
        print(f"  Pages: {result['page_count']}")
        print(f"  Words: {result['word_count']}")
        print(f"  Time: {result['processing_time_seconds']}s")
        print(f"\n  Summary preview:")
        print(f"  {result['summary'][:200]}...")
        break
    elif status_data['status'] == 'FAILURE':
        print(f"✗ Failed: {status_data.get('error')}")
        break
    elif attempt % 5 == 0:
        print(f"  Still processing... ({status_data['status']})")
    
    time.sleep(3)
```

## Architecture Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP POST /process
       ▼
┌─────────────────────┐
│   AsyncPDFAPI       │ ← Ingress Deployment
│ (HTTP Endpoints)    │
└──────┬──────────────┘
       │ enqueue_task()
       ▼
┌─────────────────────┐
│   Redis Queue       │ ← Message Broker
│ (Celery Backend)    │
└──────┬──────────────┘
       │ consume tasks
       ▼
┌─────────────────────┐
│   PDFProcessor      │ ← Task Consumer Deployment
│ @task_consumer      │   (Scaled to N replicas)
│ - process_pdf       │
└─────────────────────┘
```

## Key Concepts

### Task Consumer

The `@task_consumer` decorator transforms a Ray Serve deployment into a Celery worker that processes tasks from a queue:

```python
@serve.deployment(num_replicas=2, max_ongoing_requests=5)
@task_consumer(
    TaskProcessorConfig(
        queue_name="pdf_processing_queue",
        adapter_config=CeleryAdapterConfig(...),
        max_retries=3,
    )
)
class PDFProcessor:
    ...
```

### Task Handler

The `@task_handler` decorator marks a method that processes a specific task type:

```python
@task_handler(name="process_pdf")
def process_pdf(self, pdf_url: str, max_summary_paragraphs: int = 3):
    # Download PDF, extract text, generate summary
    return {"status": "success", ...}
```

### Task Adapter

The adapter provides methods to interact with the task queue:

```python
# Enqueue a task
task_id = adapter.enqueue_task_sync(
    task_name="process_pdf",
    kwargs={"pdf_url": url}
)

# Check status
status = adapter.get_task_status_sync(task_id)
```

## Deploy to Anyscale

1. Update Redis configuration in `server.py` with your production Redis instance
2. Deploy using the Anyscale CLI:

```bash
anyscale service deploy -f service.yaml
```

3. Get your service URL:

```bash
anyscale service status
```

## Learn More

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [Asynchronous Inference Guide](https://docs.ray.io/en/master/serve/asynchronous-inference.html)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Documentation](https://redis.io/docs/)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- [Anyscale Platform](https://docs.anyscale.com/)
