import json
import demjson3
import requests
import logging
import time
import re
from typing import List
from tqdm import tqdm
from urllib.parse import urlparse
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load the JSON file
with open("knowledge_base_deepseek_distilled.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

# Convert vulnerabilities into Document objects
raw_docs = []
for vuln_id, vuln in kb["vulnerability_mapping"].items():
    text = f"""Name: {vuln['name']}
Description: {vuln['description']}

Standard Payloads:
{chr(10).join(vuln['payload_templates'].get('standard', []))}

Bypass Payloads:
{chr(10).join(vuln['payload_templates'].get('bypass', []))}
"""

    metadata = {
        "id": vuln_id,
        "vuln_name": vuln["name"],
        "affected_endpoints": vuln.get("affected_endpoints", []),
        "target_parameters": vuln.get("target_parameters", []),
        "category": "vulnerability",
    }

    raw_docs.append(Document(page_content=text, metadata=metadata))

# Chunking the documents for better retrieval
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],  # fallback splitting
)

docs = splitter.split_documents(raw_docs)

# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(docs, embeddings)

# Save the FAISS index
faiss_index.save_local("owasp_faiss_index")

# (Optional) Load the FAISS index later like this:
# faiss_index = FAISS.load_local("owasp_faiss_index", embeddings, allow_dangerous_deserialization=True)


# with open("knowledge_base_deepseek_distilled.json", "r", encoding="utf-8") as f:
#     kb = json.load(f)

# docs = []
# for vuln in kb["vulnerability_mapping"].values():
#     text = f"""Name: {vuln['name']}
# Description: {vuln['description']}
# Standard Payloads:\n{chr(10).join(vuln['payload_templates'].get('standard', []))}
# Bypass Payloads:\n{chr(10).join(vuln['payload_templates'].get('bypass', []))}
# """
#     docs.append(Document(page_content=text,metadata={
#         "vuln_name": vuln["name"],
#         "affected_endpoints": vuln.get("affected_endpoints", []),
#         "target_parameters": vuln.get("target_parameters", []),
#         "category": "vulnerability",
#     }))



# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# faiss_index = FAISS.from_documents(docs, embeddings)
# faiss_index.save_local("owasp_faiss_index")
# faiss_index = FAISS.load_local("owasp_faiss_index", embeddings,allow_dangerous_deserialization=True)


def fetch_relevant_vulns(query_text,top_k=3):
    matches = faiss_index.similarity_search(query_text, k=top_k)
    return "\n\n".join([match.page_content for match in matches])

def match_llm_url(api_url, llm_output_keys):
    # Parse the API URL and extract just the path
    parsed_input = urlparse(api_url)
    input_path = parsed_input.path
    
    # If the path is empty, use the full URL as a fallback
    if not input_path:
        input_path = api_url
    
    # For each key in the LLM output
    for key in llm_output_keys:
        # If the key is just a path (starts with /)
        if key.startswith('/'):
            # Compare the input path with this key
            if input_path.endswith(key) or key.endswith(input_path):
                return key
        else:
            # If the key is a full URL, parse it
            parsed_key = urlparse(key)
            key_path = parsed_key.path
            
            # Compare paths
            if input_path.endswith(key_path) or key_path.endswith(input_path):
                return key
    
    # Fallback: Try a substring match if no exact match is found
    for key in llm_output_keys:
        if key in api_url or api_url.endswith(key.lstrip('/')):
            return key
    
    return None

# Configure logging
log_filename = "payload_generation.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log(message, level="info"):
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    print(message)

# Function to parse JSON file into a dictionary
def parse_json_to_dict(json_filename):
    with open(json_filename, "r", encoding="utf-8") as file:
        apis = json.load(file)

    parsed_apis = []
    for api in apis:
        parsed_apis.append({
            "name": api.get("name"),
            "method": api.get("method"),
            "url": api.get("url"),
            "headers": api.get("headers", {}),
            "body": api.get("body", ""),
            "response": {
                "status_code": api.get("response", {}).get("status_code"),
                "headers": api.get("response", {}).get("headers", {}),
                "body": api.get("response", {}).get("body", "")
            }
        })
    return parsed_apis

def extract_parameters_from_request(api_dict: dict) -> List[str]:
    """
    Given an API dict from Postman or Swagger, extract query/body parameters.
    """
    params = []

    # Extract from query_params
    query_params = api_dict.get("query_params", {})
    if isinstance(query_params, dict):
        params.extend(list(query_params.keys()))

    # Extract from body (if JSON)
    body = api_dict.get("body", {})
    if isinstance(body, dict):
        params.extend(list(body.keys()))
    
    return list(set(params))  # remove duplicates

# LMStudio API settings
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
def robust_llm_json_parser(payload_text: str, debug: bool = False):
    """Robustly parses LLM-generated JSON"""
    original = payload_text

    try:
        # Pre-process step: Convert the entire payload to a more normalized format
        # Handle specific pattern with newline + comma
        if ",\n" in payload_text or "\n," in payload_text:
            # First, convert all newlines to spaces to simplify
            payload_text = re.sub(r'\s*\n\s*', ' ', payload_text)
            
            # Then fix the specific pattern we're seeing (value comma key)
            payload_text = re.sub(r'"\s+,\s+"', '", "', payload_text)
            payload_text = re.sub(r'}\s+,\s+{', '}, {', payload_text)
            
            # Also normalize any JSON formatting issues
            payload_text = re.sub(r':\s+"', ': "', payload_text)
            
            # if debug:
            #     print("After normalization:", payload_text)
        
        # Rest of your existing code...
        if payload_text.strip().startswith('[') and not payload_text.strip().endswith(']'):
            payload_text += ']'

        # 🚑 Fix: If object starts but doesn't close
        if payload_text.strip().startswith('{') and not payload_text.strip().endswith('}'):
            payload_text += '}'
        # Strip markdown formatting
        if payload_text.startswith("```json") and payload_text.endswith("```"):
            payload_text = payload_text[7:-3].strip()
            
        # Rest of the function remains the same...
        
        # Final parse with a simplified approach
        try:
            # Try with the standard JSON parser first
            parsed = json.loads(payload_text)
            return parsed
        except json.JSONDecodeError:
            # Fall back to demjson3 for more tolerant parsing
            parsed = demjson3.decode(payload_text)
            return parsed

    except Exception as e:
        if debug:
            print("\n❌ JSON Parsing Failed:", str(e))
            print("\n🔍 Raw fragment:\n", original)
        return {
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_raw_fragment": original
        }

# Extract valid JSON block from mixed LLM output
def generate_payloads(api_data,number_of_payloads):
    # Step 1: Fetch relevant vulnerability descriptions
    query = f"{api_data['method']} {api_data['url']} {api_data['body']} {api_data['headers']} {api_data['response']['status_code']}"
    retrieved_vulns = fetch_relevant_vulns(query, top_k=10)  # from the FAISS code you already set up
    params = extract_parameters_from_request(api_data)

    prompt = f"""
    
    You are an advanced API security penetration tester specializing in OWASP Top 10 API vulnerabilities. Based on the provided API request and response details, analyze potential vulnerabilities and generate realistic, high-confidence payloads only where applicable.

---


---
### 🧩 API Request & Response:

- Method: { api_data['method'] }
- URL:{ api_data['url'] }
- Headers: { json.dumps(api_data['headers']) }
- Query/Path/Body Parameters: { params }
- Body: { api_data['body'] }

---
### 🧾 API Response:

- Status Code: { api_data['response']['status_code'] }
- Headers: { json.dumps(api_data['response']['headers'], indent=2) }
- Body: { api_data['response']['body'] }

---
### 🎯 Payload Generation Rules:

- Only generate **up to { number_of_payloads } payloads** that are **realistically applicable** to the current API context. If fewer apply, return fewer.
- Each payload should:
    - Target a specific OWASP API Top 10  vulnerability that fits the API's structure and purpose.
    - Be formatted as **valid JSON** (double-quoted keys/values, no JavaScript expressions).
    - Include a clear **injection point** (e.g., `body → token`, `query → user_id`, `path → /{{id}}`, `header → X-API-Key`).
    - Contain a strong and realistic example value (especially for credentials or tokens).
    - Provide a concise **description** of what the payload exploits, how it works, and its potential impact.
    - Avoid assumptions not supported by the API's request/response (e.g., no guessing fields or endpoints)

Use strong but bounded values. For example:
- Password: "Cr4zyP@ssw0rd123!"
- JWT: "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhYmNkZUBnbWFpbC5jb20iLCJpYXQiOjE3NDYxNjUzNTQsImV4cCI6MTc0Njc3MDE1NCwicm9sZSI6InVzZXIifQ.g1LpSMY9-lxCmcyyA-xOwp5R8WpxvtLcN5GXASS6CAml8uYtUAOBQq6vyCR5NKHVEfeyvfKULaPsQlvxXtxoM4umnBrDs4v_mdYriIeRHNEl6e3d_EW5OiQrXg1Wz-rK9H6OQ3Do09zo_7f45_11RU_Bl7KLoh2cywaSJKrAn-bhbMhr730lafjTHC8bP5Ywgk1ixedluFLJP60OVxZRDJvN2jdPGYGFtglwnNzYgsbajYFexsZp0ejRFspfAe2d_0eZVq8A0AjMFS6Q0l_m1KZkXl9FZYfaRPZoQwuqu0wxKDnIHI3SBlX-MAzyJs1E8TbBVnG2EG1dOROS69kG4A"
    """

                
    try:
        response = requests.post(LMSTUDIO_URL, json={
            "model": "Pentest_AI-GGUF/pentest_ai.Q8_0.gguf",
            "messages": [{"role": "user", "content": prompt}],
})
        
      

        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                payload_text = response_json["choices"][0]["message"]["content"]

                try:
                    if payload_text.startswith("```json") and payload_text.endswith("```"):
                        payload_text = payload_text[7:-3].strip()

                    payloads = robust_llm_json_parser(payload_text,debug=True)


                    if isinstance(payloads, list):
                        # fixed_payloads = smart_process_payloads(payloads)
    # ✅ Handle list-based LLM output
                        result = {}
                        for item in payloads :
                            endpoint = item.get("endpoint")
                            vuln_type = item.get("vulnerability_type", "Unknown")
                            if not endpoint or not vuln_type:
                                continue

                            payload_info = {
                                "payload": item.get("payload"),
                                "injection_point": item.get("injection_point"),
                                "description": item.get("description", "")
                            }

                            if endpoint not in result:
                                result[endpoint] = {}
                            result[endpoint][vuln_type] = payload_info
                            
                        return result
                    elif isinstance(payloads, dict):
    # ✅ Handle dict-based LLM output (URL → vuln_type → payload)
                        # fixed_payloads = smart_process_payloads(payloads)
                        result = {api_data["url"]: {}}
                        matched_url = match_llm_url(api_data["url"], payloads.keys())

                        if matched_url:
                            for vuln_type, entry in payloads[matched_url].items():
                                if isinstance(entry, dict):
                                    result[api_data["url"]][vuln_type] = {
                                        "payload": entry.get("payload"),
                                        "injection_point": entry.get("injection_point"),
                                        "description": entry.get("description", "")
                                    }
                                elif isinstance(entry, list):
                                    result[api_data["url"]][vuln_type] = [
                                        {
                                            "payload": item.get("payload"),
                                            "injection_point": item.get("injection_point"),
                                            "description": item.get("description", "")
                                        }
                                        for item in entry if isinstance(item, dict)
                                    ]
                            return result
                       
                except demjson3.JSONDecodeError as e:
                    log(f"JSON Parsing Error for {api_data['url']}: {str(e)}", "error")
                    log(f"Raw fragment:\n{payload_text}", "error")
                    return {
                        "_error": str(e),
                        "_error_type": "JSONDecodeError",
                        "_raw_fragment": payload_text
                    }
            else:
                log(f"No choices returned from LLM for {api_data['url']}", "error")
                return {}
        else:
            log(f"Error from LMStudio: {response.status_code} - {response.text}", "error")
            return {}

    except requests.RequestException as e:
        log(f"Request error for {api_data['url']}: {str(e)}", "error")
        return {}
    with open("last_prompt.txt", "w") as f:
        f.write(prompt)

# Load and parse API request/response data
input_filename = "output1111.json"
apis = parse_json_to_dict(input_filename)
log("Loaded and parsed API request/response data from output.json")

payloads_output = {}
log("Starting payload generation...")
abc=int(input("Enter number of payloads for every API : "))
for api in tqdm(apis, desc="Processing APIs", unit="API"):
    log(f"Generating payloads for {api['url']}")
    result = generate_payloads(api, abc)
    if isinstance(result, dict):
        payloads_output.update(result)
    time.sleep(0.5)
output_filename = "test_payloads1.json"
with open(output_filename, "w", encoding="utf-8") as output_file:
    json.dump(payloads_output, output_file, indent=4, ensure_ascii=False)


log(f"✅ Payloads saved to '{output_filename}'")
log(f"✅ Log file saved as '{log_filename}'")
