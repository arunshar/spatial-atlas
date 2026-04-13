# Spatial Atlas: The "Grandma-Friendly" Tutorial

**A warm, patient, step-by-step guide to running the Spatial Atlas project from absolute zero.**

> Imagine you just got a brand-new computer and someone told you to "run this AI project." This tutorial assumes you know *nothing* about coding, terminals, or AI. We'll walk through every single click, every single word you type, and explain *why* you're doing it. Take a deep breath — you've got this!

---

## Table of Contents

1. [What Is This Thing, Anyway?](#1-what-is-this-thing-anyway)
2. [What You'll Need Before We Start](#2-what-youll-need-before-we-start)
3. [Step 1: Open Your Terminal (The Magic Black Box)](#step-1-open-your-terminal-the-magic-black-box)
4. [Step 2: Install the Tools We Need](#step-2-install-the-tools-we-need)
5. [Step 3: Download the Project](#step-3-download-the-project)
6. [Step 4: Set Up Your Secret Key](#step-4-set-up-your-secret-key)
7. [Step 5: Install the Project's Ingredients](#step-5-install-the-projects-ingredients)
8. [Step 6: Run the Tests (Make Sure Everything Works)](#step-6-run-the-tests-make-sure-everything-works)
9. [Step 7: Start the Agent Server](#step-7-start-the-agent-server)
10. [Step 8: Verify It's Working](#step-8-verify-its-working)
11. [Step 9: Stop the Server](#step-9-stop-the-server)
12. [Bonus: Run It With Docker](#bonus-run-it-with-docker)
13. [Troubleshooting: When Things Go Wrong](#troubleshooting-when-things-go-wrong)
14. [What Does Each Piece Do? (The Guided Tour)](#what-does-each-piece-do-the-guided-tour)
15. [Glossary: Big Words Made Simple](#glossary-big-words-made-simple)

---

## 1. What Is This Thing, Anyway?

Think of **Spatial Atlas** as a very smart assistant that can do two things:

**Job #1 — "Field Work Analyst" (FieldWorkArena)**
Imagine you show it photos and documents from a factory or warehouse. It can look at those pictures and answer questions like: "Are the workers wearing hard hats?", "How far apart are the shelves?", or "Are there any safety violations?" It doesn't just guess — it builds a mental map (called a "scene graph") and *calculates* the answers with math.

**Job #2 — "Data Science Robot" (MLE-Bench)**
Give it a data science competition (like ones from Kaggle, a website where data scientists compete), and it will automatically write Python code, train a machine learning model, and produce the answer file — all by itself. If its code crashes, it reads the error, fixes the code, and tries again (up to 3 times!).

It communicates using a standard protocol called **A2A** (Agent-to-Agent), which is like a common language that AI agents use to talk to each other.

---

## 2. What You'll Need Before We Start

Here's your shopping list. Don't worry — everything is free except the OpenAI API key (which costs a tiny bit of money per use):

| What | Why You Need It | How to Get It |
|------|-----------------|---------------|
| A computer | Mac, Windows, or Linux — any will do | You probably already have one! |
| Internet connection | To download things | Your Wi-Fi or ethernet cable |
| Python 3.12 or newer | The programming language the project is written in | We'll install it below |
| `uv` | A tool that installs Python packages (think of it as an app store for code) | We'll install it below |
| `git` | A tool that downloads code projects (like a photocopier for code) | Usually pre-installed on Mac/Linux |
| An OpenAI API key | The project uses OpenAI's AI models to think | Sign up at [platform.openai.com](https://platform.openai.com) |

---

## Step 1: Open Your Terminal (The Magic Black Box)

The **Terminal** is a text-based way to talk to your computer. Instead of clicking icons, you type commands. It looks like a black or white box with blinking text.

### On a Mac:
1. Press **Command + Space** on your keyboard (this opens Spotlight Search — it's the magnifying glass)
2. Type the word: `Terminal`
3. Press **Enter**
4. A window will appear with a blinking cursor. That's your Terminal!

### On Windows:
1. Press the **Windows key** on your keyboard
2. Type: `PowerShell`
3. Click **"Windows PowerShell"**
4. A blue window will appear. That's your Terminal!

### On Linux:
1. Press **Ctrl + Alt + T**
2. A terminal window will appear

> **Grandma tip:** Think of the Terminal as texting your computer. You type a message (command), press Enter, and the computer texts you back with the result.

---

## Step 2: Install the Tools We Need

Now we're going to install our tools, one by one. Type each command exactly as shown, then press **Enter** after each one.

### 2a. Check if Python is installed

Type this and press Enter:
```
python3 --version
```

**What you should see** (the exact number might be slightly different):
```
Python 3.12.4
```

**If you see an error** like "command not found":
- **Mac:** Type `brew install python@3.12` and press Enter. (If `brew` doesn't work either, first install Homebrew by going to [brew.sh](https://brew.sh) and following their one-line install command.)
- **Windows:** Go to [python.org/downloads](https://python.org/downloads), download the installer, and run it. **Important:** Check the box that says "Add Python to PATH" during installation!
- **Linux:** Type `sudo apt install python3.12` and press Enter

### 2b. Install `uv` (the package manager)

This is a tool that manages all the code libraries our project needs. Think of it as a librarian who finds and organizes all the right books.

Type this and press Enter:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**What happens:** Your computer downloads and installs `uv`. You'll see a progress bar and a success message.

After it finishes, **close your Terminal and reopen it** (so it recognizes the new tool).

Verify it worked:
```
uv --version
```

You should see something like `uv 0.6.x`.

### 2c. Check if `git` is installed

Type:
```
git --version
```

**What you should see:**
```
git version 2.x.x
```

**If you see an error:**
- **Mac:** It will ask you to install "Command Line Tools" — click "Install" and wait
- **Windows:** Go to [git-scm.com](https://git-scm.com) and download the installer
- **Linux:** Type `sudo apt install git`

> **Grandma tip:** We just installed three tools: Python (the language), uv (the librarian), and git (the photocopier). These are the only tools we need!

---

## Step 3: Download the Project

Now we're going to download the Spatial Atlas code to your computer. This is called "cloning" — think of it as photocopying the entire project folder.

### 3a. Choose where to put it

First, let's go to a nice, clean folder. Type:
```
cd ~/Desktop
```

> This takes you to your Desktop folder. The `cd` command means "change directory" (go to a folder). The `~` symbol means "my home folder."

### 3b. Download (clone) the project

Type:
```
git clone https://github.com/arunshar/spatial-atlas.git
```

**What happens:** Git connects to GitHub (a website that stores code), downloads every file in the project, and puts it in a new folder called `spatial-atlas` on your Desktop.

You'll see something like:
```
Cloning into 'spatial-atlas'...
remote: Enumerating objects: 150, done.
remote: Counting objects: 100% (150/150), done.
Receiving objects: 100% (150/150), done.
```

### 3c. Go into the project folder

Type:
```
cd spatial-atlas
```

> Now you're inside the project! Every command from here on should be typed while you're in this folder.

**To confirm you're in the right place,** type:
```
ls
```

You should see a list of files including: `README.md`, `src`, `tests`, `pyproject.toml`, `sample.env`, etc.

---

## Step 4: Set Up Your Secret Key

The project uses OpenAI's AI to think. To use OpenAI, you need an **API key** — think of it as a password that lets the project use the AI service.

### 4a. Get your OpenAI API key

1. Go to [platform.openai.com](https://platform.openai.com) in your web browser
2. Sign up for an account (or log in if you have one)
3. Click on **"API keys"** in the left sidebar
4. Click **"Create new secret key"**
5. Give it a name like "Spatial Atlas"
6. **Copy the key** — it starts with `sk-` and is very long. **Save it somewhere safe!** You won't be able to see it again.
7. You'll need to add some credits to your account (even $5 is enough to get started)

### 4b. Create your environment file

The project looks for your secret key in a file called `.env` (the dot at the beginning means it's a hidden file — like a secret note).

Type:
```
cp sample.env .env
```

> This command copies the sample file to a new file called `.env`. The `cp` command means "copy."

### 4c. Edit the file with your key

Now we need to open the `.env` file and put your real key in it.

Type:
```
nano .env
```

> `nano` is a simple text editor that works inside the Terminal. Think of it as a very basic Notepad.

You'll see something like:
```
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=sk-...
```

**What to do:**
1. Use your arrow keys to move the cursor to the line that says `OPENAI_API_KEY=sk-...`
2. Delete the `sk-...` part
3. Paste your real API key there (on Mac: **Command + V**, on Linux: **Ctrl + Shift + V**)
4. The line should now look like: `OPENAI_API_KEY=sk-proj-abc123...your-real-key-here...`

**To save and exit nano:**
1. Press **Ctrl + O** (that's the letter O, not zero) — this means "save"
2. Press **Enter** to confirm the filename
3. Press **Ctrl + X** — this means "exit"

> **Grandma tip:** You just created a secret note that tells the project your password for using the AI service. This file stays on YOUR computer and is never shared with anyone.

---

## Step 5: Install the Project's Ingredients

Every coding project needs "dependencies" — other pieces of code that it uses. Think of dependencies like ingredients in a recipe. You need flour, eggs, and sugar before you can bake a cake.

Type:
```
uv sync
```

**What happens:** The `uv` tool reads the project's recipe book (`pyproject.toml`) and downloads all 93 ingredients (packages) from the internet.

You'll see something like:
```
Resolved 93 packages in 3ms
Installed 93 packages in 2s
 + a2a-sdk==0.3.25
 + litellm==1.83.0
 + numpy==2.4.4
 + pandas==3.0.2
 + scikit-learn==1.8.0
 ... (and many more)
```

If you also want to run the tests (recommended!), type:
```
uv sync --extra test
```

> This installs extra testing ingredients (like `pytest`).

**This might take a minute or two** — it's downloading a lot of code. Be patient!

---

## Step 6: Run the Tests (Make Sure Everything Works)

Before we turn on the real thing, let's make sure all the parts are working correctly. This is like checking that all the lights work before you open a store.

Type:
```
uv run pytest -v
```

> `pytest` is a testing tool. The `-v` flag means "verbose" — show me the details.

**What you should see:**
```
============================= test session starts ==============================
platform linux -- Python 3.14.3, pytest-9.0.3

tests/test_agent.py::TestDomainClassification::test_fieldwork_detection PASSED
tests/test_agent.py::TestDomainClassification::test_mlebench_detection_by_file PASSED
tests/test_agent.py::TestDomainClassification::test_mlebench_detection_by_keyword PASSED
tests/test_agent.py::TestDomainClassification::test_default_to_fieldwork PASSED
tests/test_agent.py::TestGoalParser::test_standard_goal PASSED
tests/test_agent.py::TestGoalParser::test_missing_sections PASSED
tests/test_agent.py::TestAnswerFormatter::test_format_numeric PASSED
tests/test_agent.py::TestAnswerFormatter::test_format_boolean PASSED
tests/test_agent.py::TestAnswerFormatter::test_format_json PASSED
tests/test_agent.py::TestAnswerFormatter::test_format_json_extraction PASSED
tests/test_agent.py::TestAnswerFormatter::test_format_list PASSED
tests/test_agent.py::TestAnswerFormatter::test_strip_markdown PASSED
tests/test_agent.py::TestAnswerFormatter::test_passthrough PASSED
tests/test_agent.py::TestSpatialScene::test_distance_computation PASSED
tests/test_agent.py::TestSpatialScene::test_query_near PASSED
tests/test_agent.py::TestSpatialScene::test_ppe_violation PASSED
tests/test_agent.py::TestSpatialScene::test_fact_sheet PASSED
tests/test_agent.py::TestSpatialScene::test_empty_scene PASSED
tests/test_agent.py::TestCostTracker::test_tracker_init PASSED
tests/test_agent.py::TestCostTracker::test_budget_exceeded PASSED
tests/test_agent.py::TestConfig::test_default_config PASSED

============================== 21 passed in 0.87s ==============================
```

**The magic words you're looking for:** `21 passed` in green. If you see this, everything is working perfectly!

> **Grandma tip:** Each "PASSED" is like a checkmark on a checklist. All 21 checkmarks mean the project is healthy and ready to run. Here's what each group tested:
> - **DomainClassification** — Can the agent figure out if it's a factory-inspection task or a data-science task?
> - **GoalParser** — Can it understand the question it's being asked?
> - **AnswerFormatter** — Can it format its answers correctly (numbers, yes/no, JSON, lists)?
> - **SpatialScene** — Can it do math with spatial locations (distances, nearby objects, safety gear)?
> - **CostTracker** — Can it keep track of how much AI usage it's spending?
> - **Config** — Are all the settings loaded correctly?

---

## Step 7: Start the Agent Server

This is the moment! We're going to turn on the Spatial Atlas agent.

Type:
```
uv run src/server.py --host 127.0.0.1 --port 9019
```

Let's break that command down for you:
- `uv run` — "Hey uv, please run this Python file for me"
- `src/server.py` — The main file that starts the agent
- `--host 127.0.0.1` — "Only listen on my own computer" (127.0.0.1 is your computer's address to itself, like looking in a mirror)
- `--port 9019` — "Use door number 9019" (ports are like numbered doors that programs use to communicate)

**What you should see:**
```
============================================================
Spatial Atlas — Purple Agent
============================================================
Server: http://127.0.0.1:9019/
Agent Card: http://127.0.0.1:9019/

Skills:
  - Multimodal Field Research: Analyzes factory, warehouse, and retail...
  - ML Engineering: Solves Kaggle-style ML competitions end-to-end...
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:9019
```

**The server is now running!** It's sitting there, patiently waiting for someone (or another AI agent) to send it a task.

> **Grandma tip:** Your computer is now acting as a little hotel that's open for business at "address 127.0.0.1, room 9019." The agent is the concierge, waiting for guests (requests) to arrive. **The Terminal will look like it's "frozen" — that's normal!** The server is running and listening. Don't close this window!

---

## Step 8: Verify It's Working

While the server is still running (don't close that Terminal window!), let's check that it's actually alive and responding.

### 8a. Open a NEW Terminal window

- **Mac:** Press **Command + N** while the Terminal app is focused, or press **Command + Space**, type Terminal, and press Enter
- **Windows:** Right-click PowerShell in the taskbar and choose "New Window"
- **Linux:** Press **Ctrl + Alt + T** again

### 8b. Ask the server to identify itself

In the new Terminal window, type:
```
curl -s http://localhost:9019/.well-known/agent-card.json | python3 -m json.tool
```

Let's break this down:
- `curl` — A tool that fetches web pages from the command line (like a tiny web browser)
- `-s` — "Be silent, don't show progress bars"
- `http://localhost:9019` — "Talk to the server running on my own computer at door 9019"
- `/.well-known/agent-card.json` — "Show me your ID card" (this is a standard endpoint for A2A agents)
- `| python3 -m json.tool` — "Make the output pretty and readable"

**What you should see:**
```json
{
    "name": "Spatial Atlas",
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
    "capabilities": {
        "streaming": true
    },
    "skills": [
        {
            "id": "fieldwork-research",
            "name": "Multimodal Field Research",
            "description": "Analyzes factory, warehouse, and retail environments...",
            "tags": ["spatial", "multimodal", "vision", "fieldwork", "research"]
        },
        {
            "id": "ml-engineering",
            "name": "ML Engineering",
            "description": "Solves Kaggle-style ML competitions end-to-end...",
            "tags": ["ml", "kaggle", "data-science", "code-generation"]
        }
    ]
}
```

**If you see this, congratulations! Your Spatial Atlas agent is alive, running, and ready to accept tasks!**

> **Grandma tip:** You just asked the server "Who are you?" and it replied with its name, version, and a list of its skills. It's like calling a restaurant and hearing their automated message: "Welcome to Spatial Atlas! We can analyze factory photos or solve data science competitions."

---

## Step 9: Stop the Server

When you're done and want to turn off the server:

1. Click on the **first** Terminal window (the one where the server is running)
2. Press **Ctrl + C** on your keyboard

> `Ctrl + C` is the universal "stop" button in the Terminal. The server will gracefully shut down.

You'll see:
```
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process
```

That's it — the server has stopped. You can close all Terminal windows now.

---

## Bonus: Run It With Docker

Docker is like a shipping container for software. It packages the entire project — code, tools, settings, and all — into a single box that runs identically on any computer. If you want to deploy this on a server or share it, Docker is the way to go.

### Install Docker first

Go to [docker.com/get-started](https://docker.com/get-started) and download Docker Desktop for your operating system. Install it, and make sure it's running (you'll see a whale icon in your system tray).

### Build the container

In your Terminal, make sure you're in the project folder, then type:
```
docker build -t spatial-atlas --platform linux/amd64 .
```

Breaking this down:
- `docker build` — "Build me a container"
- `-t spatial-atlas` — "Name it 'spatial-atlas'"
- `--platform linux/amd64` — "Build it for standard Linux servers" (required for the AgentBeats competition platform)
- `.` — "Use the current folder as the source"

**This will take a few minutes** the first time (it's downloading a base operating system image and installing everything inside the container).

### Run the container

```
docker run -p 9019:9019 --env-file .env spatial-atlas --host 0.0.0.0
```

Breaking this down:
- `docker run` — "Start the container"
- `-p 9019:9019` — "Connect my computer's door 9019 to the container's door 9019"
- `--env-file .env` — "Pass my secret API key into the container"
- `spatial-atlas` — "Use the container we just built"
- `--host 0.0.0.0` — "Listen on all network addresses" (needed inside Docker)

### Verify it (same as before)

```
curl -s http://localhost:9019/.well-known/agent-card.json | python3 -m json.tool
```

---

## Troubleshooting: When Things Go Wrong

Don't panic! Here are the most common problems and their solutions:

### "ModuleNotFoundError: No module named 'litellm'"
**What it means:** The dependencies (ingredients) aren't installed yet.
**Fix:** Run `uv sync` again.

### "OPENAI_API_KEY not set" or "AuthenticationError"
**What it means:** The project can't find your OpenAI key.
**Fix:** Make sure you created the `.env` file (Step 4). Double-check that your key is correct and doesn't have extra spaces. Re-do Step 4.

### "Address already in use" or "Port 9019 in use"
**What it means:** Something else is already using door 9019.
**Fix:** Use a different port number: `uv run src/server.py --port 9020`

### "Method Not Allowed" when opening localhost:9019 in browser
**What it means:** This is normal! The server doesn't serve web pages. It speaks the A2A protocol, not HTML.
**Fix:** Check `http://localhost:9019/.well-known/agent-card.json` instead (or use `curl` as shown in Step 8).

### Tests fail with import errors
**What it means:** You're running pytest from the wrong folder.
**Fix:** Make sure you're inside the `spatial-atlas` folder (`cd spatial-atlas`) and use `uv run pytest -v` (not just `pytest`).

### "command not found: uv"
**What it means:** uv isn't installed or your Terminal doesn't know where to find it.
**Fix:** Close your Terminal, reopen it, and try again. If that doesn't work, reinstall uv (Step 2b).

### Docker build fails
**What it means:** Usually a missing `uv.lock` file or Docker isn't running.
**Fix:** Make sure Docker Desktop is running (look for the whale icon). If `uv.lock` is missing, run `uv lock` first.

---

## What Does Each Piece Do? (The Guided Tour)

Let's take a tour of the project, file by file. Think of it as a tour of a factory where our AI agent is built.

### The Front Door: `src/server.py`
This is where everything starts. When you run the server, this file sets up the "hotel" (web server) and tells visitors what services are available. It creates an "Agent Card" — like a business card that says "Hi, I'm Spatial Atlas, and I can do field research and ML engineering."

### The Receptionist: `src/executor.py`
When a request comes in through the A2A protocol, the Executor receives it. It checks: "Is this a new task? Is this task already done?" Then it creates an Agent to handle the work and monitors progress.

### The Brain: `src/agent.py`
This is the core orchestrator. It receives the message, looks at its contents, and asks: "Is this a factory-inspection question (FieldWork) or a data-science competition (MLE-Bench)?" Then it routes the request to the right specialist team.

### The Settings File: `src/config.py`
All the configurable knobs in one place: which AI models to use, how many retries to allow, how long to let code run, etc. Think of it as the thermostat panel for the whole system.

### The AI Connector: `src/llm.py`
This is the phone line to the AI. When any part of the system needs to "think" using an AI model, it calls this file. It supports text generation, JSON output, and image analysis. It also tracks how much each call costs.

### The Cost Department: `src/cost/`
- **`tracker.py`** — Keeps a running tally of tokens used and money spent, like a utility meter.
- **`router.py`** — Decides which AI model to use based on the task. Simple tasks get the cheap/fast model. Complex spatial reasoning gets the expensive/powerful one. This saves money!

### The Entropy Engine: `src/entropy/engine.py`
This is the "confidence checker." After the agent produces an answer, the entropy engine asks: "How sure are we about this?" If confidence is below 60%, it triggers a reflection round — the agent re-examines its reasoning and tries again. Think of it as a teacher double-checking their own work.

### The Field Work Team: `src/fieldwork/`
This whole folder handles factory/warehouse/retail analysis:
- **`handler.py`** — The team leader who orchestrates the pipeline
- **`parser.py`** — Reads and understands the question being asked
- **`vision.py`** — Looks at images, reads PDFs, and extracts frames from videos
- **`detector.py`** — Uses computer vision to detect objects (people, hard hats, safety vests)
- **`spatial.py`** — Builds a map of where everything is and computes distances with real math
- **`reasoner.py`** — Combines all the evidence and formulates an answer
- **`formatter.py`** — Makes sure the answer is in the exact format expected (JSON, number, yes/no, etc.)

### The Data Science Team: `src/mlebench/`
This folder handles Kaggle-style competitions:
- **`handler.py`** — The team leader who runs the end-to-end pipeline
- **`analyzer.py`** — Looks at the competition data and figures out what kind of problem it is (classification? regression? NLP? vision? time series?)
- **`codegen.py`** — Writes complete Python scripts to solve the competition
- **`executor.py`** — Runs the generated code in a safe sandbox with a timeout
- **`strategies/`** — Pre-built templates for different types of ML problems (XGBoost for tabular data, TF-IDF for text, etc.)

---

## Glossary: Big Words Made Simple

| Term | Simple Explanation |
|------|-------------------|
| **A2A Protocol** | A standard language that AI agents use to talk to each other, like how all phones use the same calling standard |
| **API Key** | A password that lets you use an online service (like OpenAI) |
| **Agent** | A piece of software that can receive tasks and complete them autonomously |
| **Clone (git)** | Making a copy of a project from the internet to your computer |
| **Curl** | A command-line tool for fetching web pages — like a tiny invisible web browser |
| **Dependencies** | Other code libraries that a project needs to work — like ingredients in a recipe |
| **Docker** | A way to package software in a "container" so it runs the same everywhere |
| **Endpoint** | A specific URL where a server listens for requests |
| **Entropy** | A measure of uncertainty — how unsure the AI is about its answer |
| **Git** | A tool for downloading and tracking changes in code projects |
| **JSON** | A way of formatting data that computers can easily read, using curly braces `{}` |
| **Kaggle** | A website where data scientists compete to solve problems with data |
| **LiteLLM** | A library that lets you talk to many different AI providers (OpenAI, Anthropic, etc.) through one interface |
| **MLE-Bench** | A collection of 75 Kaggle competitions used to test AI coding agents |
| **Model Tier** | Different AI models for different needs: "fast" (cheap, simple tasks), "standard" (normal tasks), "strong" (hard tasks) |
| **Port** | A numbered "door" that programs use to communicate on a network |
| **Purple Agent** | An agent that handles both benchmarks (FieldWork + MLE-Bench) |
| **Pytest** | A tool for testing Python code — it runs checks to make sure everything works |
| **Scene Graph** | A structured map of objects and their relationships in a space |
| **Server** | A program that runs continuously, waiting for requests to process |
| **Terminal** | The text-based interface where you type commands to your computer |
| **UV** | A fast Python package manager — it installs and manages code libraries |
| **.env File** | A hidden file that stores secret configuration like API keys |

---

## Quick Reference Card

When you need to do it again tomorrow, here's the cheat sheet:

```bash
# Go to the project
cd ~/Desktop/spatial-atlas

# Start the server
uv run src/server.py --host 127.0.0.1 --port 9019

# (In another terminal) Check it's alive
curl -s http://localhost:9019/.well-known/agent-card.json | python3 -m json.tool

# Run the tests
uv run pytest -v

# Stop the server
# Press Ctrl+C in the server terminal
```

---

*You did it! You just ran a competition-grade AI research agent. Not bad for someone who "doesn't know computers," right?*
