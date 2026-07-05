---
name: Security issue
about: Do NOT use this template for security issues
title: 'security: DO NOT FILE PUBLICLY'
labels: security
---

## Do not file security issues here

Voidface security issues (attack pipelines that defeat protection,
weight-load path exploits, tool-side vulnerabilities) go through
the coordinated disclosure process documented in
`Documentation/process/security.md`.

Please email the security contact directly instead of opening a
public issue.

## What counts as a security issue

- A concrete attack pipeline that reliably strips Voidface
  protection and recovers the target identity.
- A tool-side bug that leaks user photos off the local machine.
- A dependency vulnerability that affects the shipped runtime.

Everything else (unexpected but non-security behavior, feature
requests, performance regressions) can go in a normal bug report
or feature request.
