## DrSR: LLM-based Symbolic Regression

Discover mathematical equations directly from data using large language models. DrSR combines LLM's scientific reasoning with gradient‑free optimization to uncover interpretable symbolic expressions.

### Quick Start

```python
from tnlearn import LLMSymRegressor
import numpy as np

# Generate sample data
X = np.random.randn(200, 2)
y = 3*X[:,0] + 0.5*np.sin(X[:,1]) + 0.1*np.random.randn(200)

# Configure LLM (supporting deepseek, siliconflow, ollama, etc.)
llm_config = {'model': 'deepseek/deepseek-chat'}
# Add environment variables for the API key:
# export DEEPSEEK_API_KEY=<your_api_key>

# Train
reg = LLMSymRegressor(llm_config=llm_config, max_iterations=5)
reg.fit(X, y)

# View discovered equation
print(reg.best_equation_)   # e.g., "return params[0]*x0 + params[1]*np.sin(x1) + params[2]"

# Predict
y_pred = reg.predict(X)
```

### Supported LLM Providers

| Provider | Environment Variable | Example `model` |
|----------|---------------------|-----------------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| SiliconFlow | `SILICONFLOW_API_KEY` | `siliconflow/Qwen/Qwen3-8B` |
| Ollama (local) | – | `ollama/llama3.1:8b` |
| BLT | `BLT_API_KEY` | `blt/gpt-4` |
| CSTCloud | `CSTCLOUD_API_KEY` | `cstcloud/gpt-oss-120b` |

## Running the Example

After installing tnlearn (e.g.,`cd tnlearn` and `pip install -e .`), you can run the DrSR example using:

```bash
python -m examples.example_drsr
```