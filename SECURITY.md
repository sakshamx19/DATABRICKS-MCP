# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately through
GitHub's "Report a vulnerability" (Security Advisories) on this repository. Include steps to
reproduce, the affected version, and the impact. We aim to acknowledge reports within 5 working
days.

## Scope

In scope:

- Bypasses of the safety layer: read-only mode, blocked safety levels, two-step confirmation,
  production protection, path allowlists.
- Secret leakage through tool responses, errors or logs.
- Path traversal or access outside configured roots.
- SQL injection through tool-built statements.

Out of scope:

- Actions the authenticated Databricks principal is legitimately allowed to perform once a user
  has confirmed them.
- The lexical SQL classifier failing to flag a statement as dangerous. It is documented as a
  guardrail, not a security boundary; real enforcement is Databricks permissions. Reports that
  improve it are still welcome.

## Deployment guidance

- Use a dedicated service principal with least privilege, not a personal admin token.
- Prefer `stdio`. If you use the HTTP transports, keep the loopback bind or put them behind an
  authenticating proxy.
- Consider `DBX_MCP_READ_ONLY=true`, or blocking `DESTRUCTIVE` / `SECURITY_SENSITIVE`, for
  general-purpose assistants.
- Never commit `.env` files.
