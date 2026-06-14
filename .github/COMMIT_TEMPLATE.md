# Commit Message Template
# ─────────────────────────────────────────────────────────────────────────────
# Format: <type>(<scope>): <subject>
#
# <body>
#
# <footer>
# ─────────────────────────────────────────────────────────────────────────────

# Type (required): What kind of change is this?
#   note:     Add a new page or section of notes
#   code:     Add a new code example, script, or notebook
#   update:   Expand or revise existing notes or code
#   fix:      Correct an error (typo, wrong formula, broken code, bad link)
#   refactor: Restructure or reorganize content without changing meaning
#   style:    Visual/formatting changes (CSS, templates, markdown formatting)
#   chore:    Maintenance tasks (dependencies, build config, indexer)
#   ci:       CI/CD configuration changes
#   feat:     New site capability (new component, plugin, or tooling)

# Scope (optional): Topic area or infra layer
#   Content:  ml, dl, nlp, cv, sys, hw, interview
#   Infra:    infra, deps, demo

# Subject (required): Short description
#   - Use imperative mood ("add" not "added" or "adds")
#   - Don't capitalize first letter
#   - No period at the end
#   - Max 50 characters

# Body (optional): Detailed explanation
#   - Wrap at 72 characters
#   - Explain what and why, not how
#   - Use bullet points

# Footer (optional): References
#   - Reference issues: "Relates to #123"

# ─────────────────────────────────────────────────────────────────────────────
# Examples:
#
#   note(nlp): add scaled-dot-product and multi-head attention pages
#
#   code(dl): add training loop example with AMP and gradient clipping
#
#   update(dl): expand training loop reference with mixed precision
#
#   - Added AMP scaler usage and gradient clipping examples.
#   - Covers torch.autocast and GradScaler patterns.
#
#   fix(nlp): correct softmax scaling factor in attention formula
#
#   refactor(nlp): reorganize attentions section with numeric prefixes
#
#   chore(infra): update indexer to preserve subdir index content
#
#   style(infra): add custom CSS for admonition callouts
# ─────────────────────────────────────────────────────────────────────────────
