# Model config

## Overview

Each model fragment under `config/model/<name>.yaml` defines the
**answer**, **judge**, and **embedding** models for a run. The answer
model and the judge model are independent and can come from different
providers (e.g. Claude for answering, Gemini for judging), so configure each
one separately in the fragment.

## Supported providers

| Model makers | API Backend | Configuring Auth |
| :--- | :--- | :--- |
| **Google** | GenAI | set env variable `GENAI_API_KEYS` |
| | GenAI | **OR** pass `--use_vertex --gcp_project_locations project_id_1:location_1,project_id_2:location_2` |
| **Anthropic** | VertexAnthropic | pass `--use_vertex --gcp_project_locations project_id_1:location_1,project_id_2:location_2` |
| **OpenAI** | LiteLLM | set env variable `OPENAI_API_KEYS` |

Backend routing is by model-name prefix (`claude-*` → Anthropic via Vertex,
`openai/*` or `gpt-*` → LiteLLM, `google/*` or `gemini-*` → GenAI), so adding a new
model is a matter of dropping a new YAML into `config/model/`.

Key rotation applies to the comma-separated key env vars used by the public
API providers:

- For Gemini, `export GENAI_API_KEYS="genai_key1,genai_key2,genai_key3"` 
- For OpenAI, `export OPENAI_API_KEYS="openai-key-1,openai-key-2"` 

## Setting thinking config

The `thinking_config` field in a model fragment controls how "thinking mode" is applied. Setting this to `null` means "do not send any thinking-related parameter", so you get whatever the provider's server-side default is. Value requirements vary significantly by provider: some models categorize thinking intensity using levels such as `"high"`, `"medium"`, `"minimal"`, while other models requires a specific maximum thinking token counts. To determine the correct format, you should consult the specific provider’s documentation. Alternatively, you can test a value directly; if it is incompatible, the system will return an error message containing suggestions for the correct format.


## Enabling Models in Vertex AI

To enable Claude models in Vertex:

1. Go to your project in [Pantheon](https://console.cloud.google.com/).
2. Ensure the **Vertex AI API** is enabled by searching for "Vertex AI API" in the API library.
3. Navigate to **Model Garden** (search for it in the console).
4. Search for the relevant model (e.g., `claude-opus-4-6`) and click **Enable**.
5. Fill out the enablement form using your credentials.

