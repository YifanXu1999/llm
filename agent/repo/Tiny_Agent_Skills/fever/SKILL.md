# ReAct Pattern for Fact Verification

## Overview
This skill enables fact verification through iterative reasoning and action. The model alternates between THOUGHT and ACTION steps to verify claims against Wikipedia knowledge.

## Core Pattern
```
Thought: [reasoning about what to do next]
Action: [one of: search[entity] | lookup[string] | finish[answer]]
Observation: [result from action]
```

## Workflow

1. **Read the claim** - Understand what needs verification
2. **Select appropriate subskill** based on current state:
   - `rules/initial.md` - First step: identify key entity to search
   - `rules/search.md` - After search: analyze results and plan next action
   - `rules/lookup.md` - During lookup: extract relevant facts
   - `rules/finish.md` - When ready: determine final answer

3. **Execute action** - Perform exactly one action per turn
4. **Process observation** - Update understanding based on result
5. **Repeat** until finish[answer] is called

## Action Constraints
- **search[entity]**: Returns first 5 sentences OR suggests similar entities
- **lookup[string]**: Returns next sentence containing the string
- **finish[answer]**: Must be SUPPORTS, REFUTES, or NOT ENOUGH INFO

## Key Principles for Small Models
- One action per turn
- Keep thoughts concise (1-2 sentences)
- Focus on claim keywords
- Maximum 5-7 steps per verification
- Stop early if answer is clear

## Selection Logic
```
Current State → Subskill
─────────────────────────
No actions yet → initial.md
After search[] → search.md
After lookup[] → lookup.md or finish.md
Ready to answer → finish.md
```

## Example Flow
```
Claim: "Albert Einstein won the Nobel Prize in Physics in 1921."

Thought: Need to verify Einstein's Nobel Prize year.
Action: search[Albert Einstein]
Observation: [Returns bio mentioning Nobel Prize]

Thought: Found Einstein page, need specific year info.
Action: lookup[Nobel Prize]
Observation: "He received the 1921 Nobel Prize in Physics..."

Thought: Claim matches observation exactly.
Action: finish[SUPPORTS]
```

## Efficiency Tips
- Search most specific entity first
- Use lookup for precise fact-finding
- Finish as soon as answer is clear
- If entity not found after 2 attempts, consider NOT ENOUGH INFO
