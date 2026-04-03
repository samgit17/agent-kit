## backend
prompt_optimizer

## goal
Improve landing page copy to score 95%+ on eval criteria. Baseline: measure first.

## target_file
skill.md

## test_inputs
- "An AI productivity tool that saves remote teams 10 hours per week by automating status updates and standup summaries"
- "A B2B SaaS CRM built specifically for marketing agencies with 10-50 employees who need pipeline visibility without enterprise complexity"
- "A personal finance app that uses AI to automatically categorize expenses and surface spending patterns you'd never catch manually"

## eval_criteria
- Does the headline include a specific number or measurable result?
- Is the copy completely free of buzzwords (revolutionary, cutting-edge, synergy, transform, empower, leverage, streamline)?
- Does the CTA use a specific action verb tied to the product outcome?
- Does the first sentence after the headline name a specific pain point?
- Is the total copy between 80 and 150 words?

## known_constraints
- Do not remove the Headline/Subheadline/Body copy/CTA structure
- Do not change the Output Format section

## constraints
outputs_per_round: 10
target_score: 0.95
max_experiments: 20
revert_on_no_improvement: true
