# Security

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**, through GitHub's private
vulnerability reporting: the repository's **Security** tab, **Report a vulnerability**
([direct link](https://github.com/Niko1221/Strata/security/advisories/new)). Say what you found, how to reproduce it
(the request, the config keys involved, the Strata version), and what an attacker gains. We answer there, fix it in
a release, and credit you in the advisory unless you ask us not to.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/Niko1221/Strata/issues) (#544 collects audit findings) or as a pull request, which we
review like our own code.

Supported: the latest release. Fixes go into the next release, not into older ones.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`). Out of the box it listens on every
interface (`0.0.0.0`), so devices on your network can reach it - set an API key, or pass `--host 127.0.0.1`, to
keep it to this PC. The details and every setting are in [docs/DETAILS.md](docs/DETAILS.md) ("From other
devices", "Host names", "Web pages without an API key").

- **Where it listens.** `0.0.0.0` by default - every interface. `--host 127.0.0.1` (or `"host"` in
  `strata-<model>.json`) keeps it to this PC. On any non-loopback address the server warns when no API key is
  set.
- **API key.** `"api_key"` in the run config (or `STRATA_API_KEY`) is required on `/v1/*` and on every endpoint
  that shows the model's state, requests or answers (`/status`, `/metrics`, `/settings`, `/props`, `/slots`,
  `/api/requests`, `/config`) and on every `POST`. It is compared in constant time. Set one before you open the
  server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding, 0.1.38).** Without an API key the server answers only requests whose `Host` is a name
  it knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's
  name). Others get 403. `"allowed_hosts"` adds names; with an API key the key decides.
- **Origin check (0.1.38).** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries
  an `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Model settings may change - never the network, key or program keys), `/load` and `/unload` are accepted
  only as JSON from Strata's own page (or `"trusted_origins"`), so a page elsewhere cannot change settings.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **Automatic tools** (claim checks, recall of earlier turns, status figures) run inside the server and start no
  program. Recall is per named session, bounded, and never crosses sessions or reaches callers without a session id;
  recalled text is put back at the user's level, quoted, never into the system message. `X-Strata-Tool-Intercept: off`
  disables them.
- **The request monitor** (`/api-monitor`, which keeps the last prompts and answers in memory) is off unless
  `"api_monitor": true` is set.

Strata has not had an outside security audit yet.
