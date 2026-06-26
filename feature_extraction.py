"""
Lexical Feature Extraction for Phishing URL Detection
-------------------------------------------------------
Extracts features purely from the URL STRING itself (no live network
requests, no WHOIS, no page content). This keeps the project self-contained
and reproducible, which is exactly what "lexical feature engineering" means
in the phishing-detection literature.
"""

import re
import math
import os
import ipaddress
from urllib.parse import urlparse
from collections import Counter

# Static list of well-known, high-traffic domains (used as a domain-reputation
# lexical signal — still derived purely from the URL string, no live lookups).
_TOP_DOMAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top_domains.txt")
try:
    with open(_TOP_DOMAINS_PATH, "r") as _f:
        TOP_DOMAINS = set(line.strip().lower() for line in _f if line.strip())
except FileNotFoundError:
    TOP_DOMAINS = set()

# Known URL-shortening services (presence is a common phishing red flag)
SHORTENERS = {
    "bit.ly", "goo.gl", "shorte.st", "go2l.ink", "x.co", "ow.ly", "t.co",
    "tinyurl.com", "tr.im", "is.gd", "cli.gs", "yfrog.com", "migre.me",
    "ff.im", "tiny.cc", "url4.eu", "twit.ac", "su.pr", "twurl.nl", "snipurl.com",
    "short.to", "budurl.com", "ping.fm", "post.ly", "just.as", "bkite.com",
    "snipr.com", "fic.kr", "loopt.us", "doiop.com", "short.ie", "kl.am",
    "wp.me", "rubyurl.com", "om.ly", "to.ly", "bit.do", "lnkd.in", "db.tt",
    "qr.ae", "adf.ly", "buff.ly", "rebrand.ly", "v.gd"
}

# Words frequently seen in phishing URLs trying to look "official"
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "bank", "webscr", "ebayisapi", "password", "credential",
    "wallet", "support", "service", "billing", "suspend", "alert", "unlock"
]


def shannon_entropy(s: str) -> float:
    """Measures how 'random-looking' a string is. Phishing URLs that use
    randomly generated subdomains/paths tend to have higher entropy."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except (ValueError, TypeError):
        return False


def get_root_domain(hostname: str) -> str:
    """Approximate the registrable root domain, e.g. 'docs.google.com' -> 'google.com'.
    This is a simple last-two-labels heuristic; it won't perfectly handle
    multi-part TLDs like .co.uk, but is good enough for a reputation lookup."""
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def count_subdomains(hostname: str) -> int:
    if not hostname:
        return 0
    parts = hostname.split(".")
    # crude but standard approximation: domain + tld take the last 2 parts
    return max(len(parts) - 2, 0)


def extract_features(url: str) -> dict:
    url = str(url).strip()

    # Ensure URL has a scheme so urlparse behaves consistently
    parse_target = url if re.match(r"^https?://", url) else "http://" + url
    parsed = urlparse(parse_target)
    hostname = parsed.netloc.split(":")[0].lower() if parsed.netloc else ""
    path = parsed.path or ""
    query = parsed.query or ""

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    words = re.split(r"[/\-_.?=&]", url)
    words = [w for w in words if w]
    word_lengths = [len(w) for w in words] if words else [0]

    features = {
        # --- Basic length features ---
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),

        # --- Character counts (special characters are common phishing cues) ---
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at": url.count("@"),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "num_digits": digits,
        "digit_ratio": round(digits / len(url), 4) if len(url) else 0,

        # --- Structural features ---
        "num_subdomains": count_subdomains(hostname),
        "has_ip_address": int(is_ip_address(hostname)),
        "is_https": int(parsed.scheme == "https"),
        "https_in_hostname": int("https" in hostname),  # deceptive use of "https" as text
        "uses_shortener": int(hostname in SHORTENERS),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),

        # --- Word-level features ---
        "num_words": len(words),
        "avg_word_length": round(sum(word_lengths) / len(word_lengths), 2),
        "longest_word_length": max(word_lengths),
        "num_suspicious_words": sum(1 for w in SUSPICIOUS_WORDS if w in url.lower()),

        # --- Domain reputation (still derived from the URL string only) ---
        "is_top_domain": int(get_root_domain(hostname) in TOP_DOMAINS),
        "top_domain_with_path": int(get_root_domain(hostname) in TOP_DOMAINS and len(path) > 1),

        # --- Randomness ---
        "url_entropy": round(shannon_entropy(url), 4),
        "hostname_entropy": round(shannon_entropy(hostname), 4),
    }
    return features


if __name__ == "__main__":
    # Quick sanity check on a few obvious examples before running on the full dataset
    samples = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/login.php?account=verify",
        "http://secure-paypa1-login.com.verify-account.xyz/signin",
        "https://bit.ly/3xampl3",
    ]
    for s in samples:
        print(s)
        print(extract_features(s))
        print()
