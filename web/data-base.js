// Public base URL of the R2 bucket `isogrid publish --r2` writes to, for the
// deployed page — e.g. "https://pub-0123456789abcdef.r2.dev". No trailing slash.
//
// Ignored on localhost, where the page always talks to `isogrid serve` at /api,
// so setting this can never break local development. To point a local copy of
// the page at published data instead, open it with ?data=<url>.
window.ISOGRID_DATA_BASE = "";
