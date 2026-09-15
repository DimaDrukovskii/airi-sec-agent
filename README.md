# AIRI Security Agent

Autonomous cybersecurity agent for the AIRI Universal Agentic Competition.

## Architecture

Task
→ Contract Extraction
→ Workflow Selection
→ Local LLM
→ Tool Execution
→ Observation
→ Verification
→ Recovery / Finish

## Capabilities

- Vulnerability analysis
- Security patching
- Digital forensics
- CTF-style tasks
- Autonomous shell and filesystem interaction
- Test-driven verification
- Runtime and request-budget management

## Runtime

The agent is designed for an isolated competition environment
using an OpenAI-compatible local LLM endpoint.

## Entrypoint

./run.sh "<task instruction>"
