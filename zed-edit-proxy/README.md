# Zed Edit Prediction Proxy

A proxy service that adapts Zed's edit prediction API format to OpenAI-compatible APIs, allowing you to use local models with vLLM, Ollama, LM Studio, or other OpenAI-compatible servers.

## Features

- **API Translation**: Converts Zed's `PredictEditsBody` format to OpenAI chat completion format
- **Multiple Backends**: Works with vLLM, Ollama, LM Studio, and any OpenAI-compatible API
- **Configurable**: Easy configuration via YAML files
- **Logging**: Comprehensive logging for debugging and monitoring
- **Health Checks**: Built-in health check endpoint

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Local Model Server

#### For LM Studio:
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a code-capable model (e.g., CodeLlama, DeepSeek Coder, etc.)
3. Start the local server (default: http://localhost:1234)

#### For Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a code model
ollama pull codellama:7b-code

# Start Ollama (it runs on http://localhost:11434 by default)
ollama serve
```

#### For vLLM:
```bash
# Install vLLM
pip install vllm

# Start vLLM server with a model
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --port 8000
```

### 3. Configure the Proxy

Edit `config.yaml` to match your setup:

```yaml
openai_base_url: "http://localhost:1234/v1"  # Your local API endpoint
model_name: "your-model-name"                # Your model name
max_tokens: 2048
temperature: 0.1
```

### 4. Start the Proxy

```bash
python main.py --config config.yaml
```

The proxy will start on `http://localhost:8080` by default.

### 5. Configure Zed

Set the environment variable to use your proxy:

```bash
export ZED_PREDICT_EDITS_URL="http://localhost:8080/predict_edits/v2"
```

Then start Zed:

```bash
zed
```

## Configuration Options

### config.yaml

| Option | Description | Default |
|--------|-------------|---------|
| `openai_base_url` | Base URL for your OpenAI-compatible API | `http://localhost:1234/v1` |
| `model_name` | Model name to use | `zed-industries/zeta` |
| `max_tokens` | Maximum tokens to generate | `2048` |
| `temperature` | Sampling temperature | `0.1` |
| `timeout` | Request timeout in seconds | `30.0` |
| `system_prompt` | System prompt for the model | See config.yaml |

### Command Line Options

```bash
python main.py [options]

Options:
  --config CONFIG    Configuration file path
  --host HOST        Host to bind to (default: 127.0.0.1)
  --port PORT        Port to bind to (default: 8080)
  --debug            Enable debug logging
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ZED_EDIT_PROXY_PORT` | Port to run on | `8080` |
| `ZED_EDIT_PROXY_HOST` | Host to bind to | `127.0.0.1` |
| `ZED_EDIT_PROXY_DEBUG` | Enable debug logging | `false` |

## API Endpoints

### POST /predict_edits/v2

Main endpoint that receives Zed's edit prediction requests and returns predictions.

**Request Format** (matches Zed's PredictEditsBody):
```json
{
  "outline": "optional string",
  "input_events": "string - user input events", 
  "input_excerpt": "string - code context around cursor",
  "speculated_output": "optional string",
  "can_collect_data": "boolean",
  "diagnostic_groups": "optional array",
  "git_info": {
    "head_sha": "optional string",
    "remote_origin_url": "optional string", 
    "remote_upstream_url": "optional string"
  }
}
```

**Response Format** (matches Zed's PredictEditsResponse):
```json
{
  "request_id": "uuid",
  "output_excerpt": "string - predicted code changes"
}
```

### POST /predict_edits/accept

Endpoint for accepting edit predictions (currently logs the acceptance).

### GET /health

Health check endpoint that tests connectivity to the OpenAI API.

## Example Configurations

### LM Studio (Default)
```yaml
openai_base_url: "http://localhost:1234/v1"
model_name: "deepseek-coder-6.7b-instruct"
```

### Ollama
```yaml
openai_base_url: "http://localhost:11434/v1" 
model_name: "codellama:7b-code"
```

### vLLM
```yaml
openai_base_url: "http://localhost:8000/v1"
model_name: "deepseek-ai/deepseek-coder-6.7b-instruct"
```

### Remote API
```yaml
openai_base_url: "https://your-remote-api.com/v1"
model_name: "your-model"
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Make sure your local model server is running
2. **Model not found**: Verify the model name matches what's available in your server
3. **Timeouts**: Increase the `timeout` value in config for slower models
4. **Poor predictions**: Adjust the `system_prompt` and `temperature` for better results

### Debug Mode

Run with `--debug` flag to see detailed request/response logs:

```bash
python main.py --config config.yaml --debug
```

### Health Check

Check if the proxy and your model server are working:

```bash
curl http://localhost:8080/health
```

## Model Recommendations

For best results with code prediction, use models specifically trained for code:

- **DeepSeek Coder** (6.7B, 33B) - Excellent for code completion
- **CodeLlama** (7B, 13B, 34B) - Good general code understanding  
- **StarCoder** (15B) - Strong at code generation
- **WizardCoder** (15B, 34B) - Good for complex code tasks

## License

This project is released into the public domain. Use it however you like!