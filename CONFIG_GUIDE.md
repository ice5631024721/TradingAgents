# TradingAgents Configuration Guide

## Overview

The TradingAgents CLI now uses a configuration file (`config.yaml`) instead of interactive command-line prompts. This allows for automated execution and easier configuration management.

## Configuration File

Create a `config.yaml` file in the project root directory with the following structure:

```yaml
# TradingAgents Configuration File
# This file contains all the settings that were previously collected through command line interaction

# Basic Analysis Settings
ticker: "SPY"  # Stock ticker symbol to analyze
analysis_date: "2025-01-27"  # Analysis date in YYYY-MM-DD format

# Analyst Team Selection
# Available options: market, social, news, fundamentals
analysts:
  - "market"
  - "social"
  - "news"
  - "fundamentals"

# Research Depth Level
# 1 = Shallow (Quick research, few debate rounds)
# 3 = Medium (Moderate debate rounds and strategy discussion)
# 5 = Deep (Comprehensive research, in-depth debate and strategy discussion)
research_depth: 3

# LLM Provider Configuration
# Available providers: openai, anthropic, google, openrouter, ollama
llm_provider: "openai"
backend_url: "https://api.openai.com/v1"

# Thinking Agents Configuration
# Shallow thinking agent for quick tasks
shallow_thinker: "gpt-4o-mini"

# Deep thinking agent for complex reasoning
deep_thinker: "o4-mini"
```

## Configuration Options

### Basic Settings
- **ticker**: Stock ticker symbol to analyze (e.g., "SPY", "AAPL", "TSLA")
- **analysis_date**: Date for analysis in YYYY-MM-DD format

### Analyst Selection
Choose from the following analyst types:
- **market**: Market Analyst
- **social**: Social Media Analyst  
- **news**: News Analyst
- **fundamentals**: Fundamentals Analyst

### Research Depth
- **1**: Shallow - Quick research, few debate rounds
- **3**: Medium - Moderate debate rounds and strategy discussion
- **5**: Deep - Comprehensive research, in-depth debate and strategy discussion

### LLM Providers

#### OpenAI
```yaml
llm_provider: "openai"
backend_url: "https://api.openai.com/v1"
shallow_thinker: "gpt-4o-mini"
deep_thinker: "o4-mini"
```

#### Anthropic
```yaml
llm_provider: "anthropic"
backend_url: "https://api.anthropic.com/"
shallow_thinker: "claude-3-5-haiku-latest"
deep_thinker: "claude-3-5-sonnet-latest"
```

#### Google
```yaml
llm_provider: "google"
backend_url: "https://generativelanguage.googleapis.com/v1"
shallow_thinker: "gemini-2.0-flash-lite"
deep_thinker: "gemini-2.0-flash"
```

#### OpenRouter
```yaml
llm_provider: "openrouter"
backend_url: "https://openrouter.ai/api/v1"
shallow_thinker: "meta-llama/llama-3.3-8b-instruct:free"
deep_thinker: "deepseek/deepseek-chat-v3-0324:free"
```

#### Ollama (Local)
```yaml
llm_provider: "ollama"
backend_url: "http://localhost:11434/v1"
shallow_thinker: "llama3.2"
deep_thinker: "qwen3:30b"
```

## Usage

1. Create your `config.yaml` file in the project root
2. Run the CLI as usual:
   ```bash
   python cli/main.py
   ```

The system will automatically load your configuration and display the loaded settings before starting the analysis.

## Migration from Interactive Mode

If you were previously using the interactive command-line interface, simply create a `config.yaml` file with your preferred settings. The configuration file provides the same functionality as the previous interactive prompts.

## Error Handling

- If `config.yaml` is not found, the system will display an error and exit
- Invalid analyst types will be ignored with a warning
- Missing configuration values will use sensible defaults
- YAML parsing errors will be displayed with helpful error messages