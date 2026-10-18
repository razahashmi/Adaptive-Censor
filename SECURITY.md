# Security policy for the preview

Only the current preview source is in scope for fixes. This is a local development service, not a hardened public or household server. It binds to `127.0.0.1`; do not expose the port through a proxy, tunnel or router.

The app processes user-selected media, runs local media codecs and can fetch explicitly requested HTTP/HLS streams. Keep the operating system and dependencies updated, use media you trust and protect the runtime directory. Browser host/origin restrictions, local cookies and write tokens reduce accidental cross-origin access; they do not replace an operating-system security boundary or user-account authentication.

For a suspected vulnerability, use the repository's private vulnerability-reporting feature if enabled. Otherwise ask a maintainer for a private reporting channel without posting exploit details or sensitive data. Do not publish credentials, copied user media, raw transcripts or private runtime files in an issue. Include a minimal generated-media reproduction where possible.

There is no guaranteed response time or security certification. A future network deployment needs a separate authentication, authorization, transport, media-access and isolation design.
