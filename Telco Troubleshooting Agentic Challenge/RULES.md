# Competition Rules

Important rules to keep in mind during experimentation and submission.

## Technical Constraints

- **Base model is fixed:** Must use Qwen3.5-35B-A3B. Fine-tuning is allowed (LoRA, full fine-tuning), but you cannot swap to a different architecture or a different parameter scale.
- **No AutoML tools** permitted.
- **Open-source tools only** — all languages, libraries, and frameworks must be open-source.
- **Provided datasets only** — no external telecom datasets may be used for training or evaluation.
- **`server.py` must not be modified** — the organizer replaces it with their own in Phase 3. Only `main.py` (and optionally fine-tuned weights) are submitted.
- **Tool schemas must be discovered dynamically** at runtime from `/tools` — do not hardcode them.
- **Data license:** CC-BY-SA 4.0 applies to all competition data.

## Deployment Requirements (Phase 3)

- **Local GPU deployment is expected** — serve Qwen3.5-35B-A3B yourself via vLLM or SGLang (OpenAI-compatible server). The competition does not provide a hosted endpoint or API key. `AGENT_API_KEY="dummy"` is a placeholder; local servers (vLLM/SGLang) ignore the Authorization header by default.
- **For Phase 1 testing**, local deployment or a commercial cloud service (e.g., OpenRouter with a real key) are both acceptable.
- **GPU + CUDA required** to run the model.
- **Separate GPU and CPU scripts** — the GPU deployment script (model serving) and the agent execution script (`main.py`) must be decoupled. The organizer runs them on different servers.
- **One-click runnable environment** — the full submission must work end-to-end without manual steps. Think startup shell scripts or a clean entry point.
- **Tool-call latency is negligible** — the organizer confirmed it will not meaningfully affect the 15-minute budget. Optimize inference time, not tool calls.

## Organizer Clarifications (from discussions)

- **All data access must go through the Agent Tool Server** — the solution must not read raw data files (e.g., `data/Phase_1/test.json`) directly. Everything must flow through the HTTP tool endpoints. In Phase 3 the JSON data files will not be available; only the server API will be. Any code that bypasses the tool server will break. *(Source: "Clarification regarding the expected solution", Apr 21)*

- **High Phase 1 scores do not guarantee generalization** — the organizers explicitly warned that Phase 1 data can be solved with pattern matching. Phase 2/3 will test deeper reasoning on the actual network data. Prioritize solutions that truly understand the telecom signals over ones that exploit Phase 1 patterns. *(Source: "Does a high score on Phase 1 mean...", Apr 21)*

## Submission Rules

- **Phase 1:** Unlimited daily result.csv submissions; max 1,000 API calls/day on the cloud server.
- **Phase 2:** Single execution run; up to 3 result.csv submissions. Top 30 advance to Phase 3.
- **Phase 3:** One attempt only. Organizer runs `main.py` on a private dataset with uniformly deployed model weights.
- **Code submission:** Top 30 finishers must submit their full code within 48 hours of request. Winners must submit winning code for reproducibility verification — rerunning must match the leaderboard score.

## Team Rules

- Maximum **4 members** per team.
- No multiple accounts per user.
- No code sharing outside your team.
- No collaboration across teams.
- Winning Track A disqualifies from Track B prizes (and vice versa) — only one prize per person/team.

## Scoring Reminders

- IoU accuracy: partial credit for overlapping answers (relevant for multi-answer questions).
- Time budget per scenario matters from Phase 3 onward — keep total solve time under 5 minutes per scenario to avoid score discounts.
- Tiebreaker is earliest submission timestamp — submit early when confident.

## Violations

- First violation: 6-month probation + 2,000 Zindi points removed.
- Second violation: permanent platform ban.
