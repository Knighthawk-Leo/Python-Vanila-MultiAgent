

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Interaction](#agent-interaction)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Example Queries](#example-queries)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Features](#features)

---

## 🌟 Overview

This project implements a **multi-agent AI system** that can:

- 📊 **Analyze CSV data** with natural language queries
- 📈 **Create visualizations** automatically based on your questions
- 💡 **Provide clear answers** without technical jargon
- 🤖 **Route intelligently** using AI-powered agent selection
- 💬 **Handle conversations** like ChatGPT

**Key Differentiators:**
- Built from **first principles** (no LangChain/LlamaIndex)
- **AI-powered routing** using Gemini for intelligent agent selection
- **Clean user experience** - users see answers, not code
- **Collaborative agents** that pass context seamlessly

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
│              (Natural Language or CSV Upload)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Orchestrator                         │
│          (Gemini AI-Powered Intelligent Routing)             │
└─────┬──────────────┬──────────────┬────────────────────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
│   Code   │  │   Viz    │  │Presentation  │  │  Answer  │
│Interpreter│ │  Agent   │  │   Agent      │  │Synthesiser│
└──────────┘  └──────────┘  └──────────────┘  └──────────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Final Answer to    │
              │   User (Markdown)    │
              └──────────────────────┘
```

### Core Components

#### 1. **Agent Orchestrator** (`orchestrator.py`)
The brain of the system that:
- Uses **Gemini AI** to intelligently route queries to the right agent
- Manages context flow between agents
- Coordinates multi-agent workflows
- Maintains conversation history

#### 2. **Code Interpreter Agent** (`code_interpreter.py`)
Handles data analysis:
- Loads and processes CSV files
- Generates Python code using Gemini
- Executes code safely in a sandboxed environment
- Performs statistical analysis and data manipulation
- Extracts insights from data

#### 3. **Visualization Agent** (`visualization_agent.py`)
Creates visual representations:
- Generates matplotlib/seaborn chart code
- Creates various chart types (bar, line, scatter, heatmap, etc.)
- Renders visualizations as base64-encoded images
- Ensures charts are clear and informative

#### 4. **Presentation Agent** (`presentation_agent.py`)
Formats reports (rarely used now):
- Generates structured reports
- Creates executive summaries
- Formats insights professionally

#### 5. **Answer Synthesiser Agent** (`answer_synthesiser.py`)
**The final agent** - provides user-friendly answers:
- Answers general questions directly
- Synthesizes results from other agents
- Formats responses in markdown
- Hides technical details from users
- Creates conversational, clear explanations

### Technology Stack

```
Backend:        FastAPI + Uvicorn
AI:             Google Gemini 2.0 Flash
Data:           Pandas, NumPy
Visualization:  Matplotlib, Seaborn
Validation:     Pydantic
Language:       Python 3.10+
```

---

## 🔄 Agent Interaction

### How Agents Work Together

The system uses **intelligent agent routing** where Gemini AI decides which agent should handle each query based on:
- User's query intent
- Current session context
- Available data
- Agent capabilities

### Interaction Patterns

#### Pattern 1: Simple Question
```
User: "What is machine learning?"
   ↓
Gemini Routes → AnswerSynthesiser
   ↓
Output: Clean explanation in markdown
```

#### Pattern 2: Data Analysis
```
User: [Uploads CSV] "Analyze sales data"
   ↓
Gemini Routes → CodeInterpreter
   ├─ Loads CSV
   ├─ Generates analysis code
   ├─ Executes code
   └─ Extracts insights
   ↓
Routes to → AnswerSynthesiser
   ├─ Synthesizes findings
   └─ Formats as user-friendly answer
   ↓
Output: "Based on your sales data, here are the key findings..."
```

#### Pattern 3: Analysis with Visualization
```
User: "Show me sales trends over time"
   ↓
Gemini Routes → CodeInterpreter
   ├─ Analyzes data
   └─ Determines visualization needed
   ↓
Routes to → VisualizationAgent
   ├─ Generates chart code
   ├─ Creates beautiful charts
   └─ Encodes as images
   ↓
Routes to → AnswerSynthesiser
   ├─ References visualizations
   └─ Explains trends
   ↓
Output: Charts + "As shown above, sales increased 25% in Q3..."
```

### Context Flow

Agents pass information through a **shared context**:

```python
context = {
    'codeinterpreter_data': {
        'analysis': "...",
        'results': [...],
        'dataframes_info': {...}
    },
    'visualizationagent_data': {
        'visualizations': [...],
        'visualization_count': 2
    }
}
```

Each agent:
1. Receives context from previous agents
2. Processes the query
3. Adds its results to context
4. Routes to next agent (if needed)

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10 or higher**
- **Google Gemini API Key** ([Get one free here](https://makersuite.google.com/app/apikey))
- **Git** (for cloning)

---

## 📝 Example Queries

### General Questions

```
"What is machine learning?"
"Explain the difference between mean and median"
"How does linear regression work?"
"What is a pandas DataFrame?"
```

**Response:** Clear explanations with examples

### Data Analysis Queries

```
"Load this CSV and show me summary statistics"
"What are the top 5 products by sales?"
"Calculate the average revenue by region"
"Find correlations between variables"
"Identify outliers in the price column"
```

**Response:** Statistical analysis with key findings

### Visualization Requests

```
"Create a bar chart of sales by category"
"Show me a line graph of monthly trends"
"Visualize the distribution of customer ages"
"Plot the correlation heatmap"
"Create a scatter plot of price vs quantity"
```

**Response:** Beautiful charts with explanations

### Combined Analysis

```
"Analyze this sales data, create visualizations, and provide insights"
"Show me revenue trends over time with a detailed breakdown"
"Compare sales across regions and highlight top performers"
```

**Response:** Complete analysis with charts and insights

### Conversational Queries

```
"Hello!"
"What can you do?"
"Thanks for the analysis"
"Can you explain that in simpler terms?"
```

**Response:** Friendly, conversational answers

### Follow-up Questions

After uploading data:
```
First: "Analyze this data"
Then: "What were the key findings?"
Then: "Show me a visualization of that"
Then: "Which region performed best?"
```

**Response:** Context-aware answers using previous analysis

---

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t multi-agent-interpreter .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  --name agent-system \
  multi-agent-interpreter

# View logs
docker logs -f agent-system

# Stop container
docker stop agent-system

# Remove container
docker rm agent-system
```

### Environment Variables

- `GEMINI_API_KEY` - Your Google Gemini API key (required)
- `PORT` - Server port (default: 8000)
