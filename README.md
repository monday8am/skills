# monday8am Skills

Agent Skills for on-device ML tasks: model fine-tuning, evaluation, and mobile deployment. Compatible with [Claude Code](https://claude.ai/code), [Codex](https://openai.com/codex), and [Gemini CLI](https://github.com/google-gemini/gemini-cli).

Skills follow the standardized [Agent Skill](https://agentskills.io/home) format.

## Available Skills

| Skill | Description |
|-------|-------------|
| [functiongemma-trainer](skills/functiongemma-trainer/) | Fine-tune Google's FunctionGemma (270M) for on-device function calling. Full pipeline: dataset validation, SFT training on HF Jobs, evaluation, and LiteRT-LM export for Android. |

## Installation

### Claude Code

1. Register the repository as a plugin marketplace:

```
/plugin marketplace add monday8am/skills
```

2. Install a skill:

```
/plugin install functiongemma-trainer@monday8am/skills
```

### Codex

Codex will identify the skills via the `agents/AGENTS.md` file:

```
codex --ask-for-approval never "Summarize the current instructions."
```

### Gemini CLI

```
gemini extensions install . --consent
```

## Links

- [FunctionGemma Trainer HF Space](https://huggingface.co/spaces/monday8am/functiongemma-trainer) — documentation landing page
- [Cycling Copilot model](https://huggingface.co/monday8am/cycling-copilot-functiongemma) — example fine-tuned model
- [Cycling Copilot dataset](https://huggingface.co/datasets/monday8am/cycling-copilot) — example training dataset
- [Project repo](https://github.com/monday8am/riding-copilot-files) — full Cycling Copilot ML pipeline

## License

MIT
