# Voidface security policy

Voidface is a defensive tool. A weakness in Voidface can put real people
in real danger. Please help us fix it quietly.

---

## What counts as a security issue

- Any concrete attack pipeline (open-source or commercial) that reliably
  strips Voidface protection and recovers the target identity or
  usable inpainting quality.
- Any bug in the training system that causes the shipped `G` to embed
  identifying information about the source image or the user.
- Any code path in the CLI, desktop app, or browser demo that leaks user
  photos off the local machine.
- Any dependency vulnerability that affects the shipped runtime.

Anything that does not fall under the above — including "the protection
level is lower than I expected" — belongs in a normal GitHub issue.

---

## How to report

Email `security@voidface.org` (placeholder pending real address).

Include:

- The vulnerability class and a one-line summary.
- A reproduction: config, seed, sample image if legal to share, the
  attack recipe, and the observed identity leakage or bypass rate.
- Whether you would like credit, and under what name.

Do **not** open a public issue for undisclosed vulnerabilities.

Do **not** attach non-consensual intimate imagery to a report, ever.
If your reproduction requires it, describe the pipeline and the identity
score numbers without attaching output.

---

## What we will do

- Acknowledge within 3 working days.
- Ship a fix within 30 days for confirmed issues, or explain the delay.
- Credit reporters in `CREDITS` unless anonymity is requested.
- Publish a post-mortem after the fix ships when the class of issue is
  general enough to be instructive.

---

## Coordinated disclosure

We ask that you give us 30 days before public disclosure. If you find an
active in-the-wild exploitation, contact us immediately and we will
prioritize accordingly.

---

## Our commitment

- No legal threats against good-faith security research.
- No demands to sign an NDA to receive a bug bounty. (There is no bug
  bounty program yet. If that changes, this file will say so.)
