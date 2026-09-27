# Security Policy

## Supported versions

Only the latest release gets security fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" on the Security tab). You should get a reply within 7 days.

## Scope notes

toolpair makes no network calls and has no runtime dependencies. It only
restructures the message lists you pass in. When you file a bug, **redact
conversation content**: the message *structure* (roles, ids, block types) is
all that's needed to reproduce a pairing problem.
