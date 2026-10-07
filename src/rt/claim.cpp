// src/rt/claim.cpp - claim detection in generated text (precision first: a false "contradiction" costs a regeneration)
#include "strata/rt/claim.hpp"
#include "strata/rt/symbolic.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <set>

namespace strata::rt {

namespace {

enum class T { kWord, kNum, kSym, kPath, kTick, kUrl };
struct Tok {
    T type = T::kSym;
    std::string s;
    size_t b = 0, e = 0;   // byte offsets within the sentence
};

bool is_word_start(unsigned char c) { return std::isalpha(c) || c == '_'; }
bool is_word_char(unsigned char c) { return std::isalnum(c) || c == '_'; }

// UTF-8 multiplication / division signs are common in model output.
bool starts_with(const std::string& s, size_t i, const char* lit) { return s.compare(i, std::strlen(lit), lit) == 0; }

std::vector<Tok> tokenize(const std::string& s) {
    std::vector<Tok> out;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isspace(c)) { ++i; continue; }
        Tok t;
        t.b = i;
        if (c == '`') {
            size_t j = s.find('`', i + 1);
            if (j == std::string::npos) j = n; else ++j;
            t.type = T::kTick; t.s = s.substr(i, j - i); i = j;
        } else if (starts_with(s, i, "://")) {
            t.type = T::kSym; t.s = "://"; i += 3;
        } else if (std::isdigit(c) || (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
            size_t j = i;
            // digits with inner , . _ separators (1,200  3.14  1.2.3  8_000); a trailing '.' ends the sentence instead
            while (j < n && (std::isdigit(static_cast<unsigned char>(s[j])) ||
                             ((s[j] == ',' || s[j] == '.' || s[j] == '_') && j + 1 < n && std::isdigit(static_cast<unsigned char>(s[j + 1])))))
                ++j;
            t.type = T::kNum; t.s = s.substr(i, j - i); i = j;
        } else if (is_word_start(c)) {
            size_t j = i;
            while (j < n && (is_word_char(static_cast<unsigned char>(s[j])) ||
                             ((s[j] == '-' || s[j] == '.') && j + 1 < n && is_word_char(static_cast<unsigned char>(s[j + 1])))))
                ++j;
            // scheme://... is a URL
            if (j + 2 < n && s.compare(j, 3, "://") == 0) {
                size_t k = j + 3;
                while (k < n && !std::isspace(static_cast<unsigned char>(s[k])) && s[k] != ')' && s[k] != ',' && s[k] != '"') ++k;
                while (k > j + 3 && (s[k - 1] == '.' || s[k - 1] == ';')) --k;
                t.type = T::kUrl; t.s = s.substr(i, k - i); i = k;
            } else {
                t.type = T::kWord; t.s = s.substr(i, j - i); i = j;
            }
        } else if ((c == '/' || (c == '~' && i + 1 < n && s[i + 1] == '/') || (c == '.' && i + 1 < n && s[i + 1] == '/')) &&
                   i + 1 < n && (is_word_char(static_cast<unsigned char>(s[i + 1])) || s[i + 1] == '.' || s[i + 1] == '/' || s[i + 1] == '~')) {
            size_t j = i + 1;
            while (j < n && !std::isspace(static_cast<unsigned char>(s[j])) && s[j] != ')' && s[j] != ',' && s[j] != '"' && s[j] != '`') ++j;
            while (j > i + 1 && (s[j - 1] == '.' || s[j - 1] == ';' || s[j - 1] == ':')) --j;
            t.type = T::kPath; t.s = s.substr(i, j - i); i = j;
        } else if (std::isalpha(c) == 0 && i + 2 < n && c < 128 && std::isalpha(static_cast<unsigned char>(s[i])) == 0 && false) {
            ++i;
        } else if (starts_with(s, i, "\xC3\x97")) { t.s = "*"; i += 2; }              // ×
        else if (starts_with(s, i, "\xC3\xB7")) { t.s = "/"; i += 2; }                 // ÷
        else if (starts_with(s, i, "\xE2\x88\x92")) { t.s = "-"; i += 3; }             // −
        else if (starts_with(s, i, "\xE2\x89\x88")) { t.s = "~="; i += 3; }            // ≈
        else if (starts_with(s, i, "==")) { t.s = "="; i += 2; }
        else if (starts_with(s, i, "**")) { t.s = "^"; i += 2; }
        else if (starts_with(s, i, "->")) { t.s = "->"; i += 2; }
        else { t.s = std::string(1, static_cast<char>(c)); ++i; }
        // A drive path C:\dir
        if (t.type == T::kWord && t.s.size() == 1 && i + 1 < n && s[i] == ':' && s[i + 1] == '\\') {
            size_t j = i + 2;
            while (j < n && !std::isspace(static_cast<unsigned char>(s[j])) && s[j] != '"' && s[j] != ')') ++j;
            while (j > i + 2 && (s[j - 1] == '.' || s[j - 1] == ';')) --j;
            t.type = T::kPath; t.s = s.substr(t.b, j - t.b); i = j;
        }
        t.e = i;
        out.push_back(std::move(t));
    }
    return out;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string strip_ticks(const std::string& s) {
    std::string r = s;
    if (r.size() >= 2 && r.front() == '`' && r.back() == '`') r = r.substr(1, r.size() - 2);
    return r;
}

// "1,200" -> "1200", "8_000" -> "8000"; decimals keep their point
std::string clean_number(const std::string& s) {
    std::string r;
    const bool has_point = std::count(s.begin(), s.end(), '.') == 1;
    for (char c : s) {
        if (c == ',' || c == '_') continue;
        r.push_back(c);
    }
    (void)has_point;
    return r;
}

const std::set<std::string>& stopwords() {
    static const std::set<std::string> w = {"is", "are", "was", "were", "be", "to", "use", "uses", "used", "the", "a", "an", "on", "of", "at",
                                            "in", "by", "for", "with", "set", "configured", "default", "defaults", "listens", "runs", "and",
                                            "or", "it", "its", "this", "that", "than", "as", "about", "around", "approximately", "roughly",
                                            "currently", "now", "only", "also", "value", "number", "equals", "equal", "has", "have", "had",
                                            "up", "max", "maximum", "min", "minimum", "from", "into", "only", "exactly", "precisely",
                                            "supports", "support", "supported", "requires", "require", "required", "provides", "allows",
                                            "handles", "includes", "contains", "using", "see", "check", "open", "edit", "named", "called",
                                            "which", "where", "when", "will", "can", "could", "should", "must", "needs", "need", "takes"};
    return w;
}

// subjects that are not "a thing with a value" (pronouns, results of a computation)
const std::set<std::string>& unusable_subjects() {
    static const std::set<std::string> w = {"i", "we", "you", "they", "he", "she", "there", "answer", "result", "total", "sum", "product",
                                            "difference", "quotient", "output", "so", "then", "thus", "therefore", "step"};
    return w;
}

// nouns that count things: "48 layers" is layers = 48
const std::set<std::string>& count_nouns() {
    static const std::set<std::string> w = {"tokens", "cores", "threads", "users", "requests", "layers", "experts", "workers", "retries",
                                            "connections", "heads", "files", "nodes", "gpus", "devices"};
    return w;
}

const std::set<std::string>& risky_subjects() {
    static const std::set<std::string> w = {"port", "timeout", "version", "size", "count", "limit", "id", "ip", "host", "address", "path",
                                            "memory", "ram", "threads", "workers", "retries", "interval", "ttl", "rate", "length", "width",
                                            "height", "depth", "price", "cost", "year", "age", "population", "speed", "latency", "capacity",
                                            "batch", "context", "tokens", "dimension", "layers", "heads", "seed", "pid", "uid"};
    return w;
}

// ---- sentence segmentation ----
bool is_abbrev(const std::string& text, size_t dot) {
    // e.g. "e.g." "i.e." "etc." "vs." "Dr." — look at the word before the dot
    size_t b = dot;
    while (b > 0 && (std::isalpha(static_cast<unsigned char>(text[b - 1])) || text[b - 1] == '.')) --b;
    std::string w = lower(text.substr(b, dot - b));
    static const std::set<std::string> ab = {"e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "fig", "eq", "no", "al", "approx", "st", "inc", "cf"};
    return ab.count(w) > 0;
}

// End offset (exclusive) of the sentence that begins at `from`, or npos if it is not complete yet.
size_t sentence_end(const std::string& text, size_t from) {
    const size_t n = text.size();
    size_t i = from;
    while (i < n) {
        // fenced code block is one unit, complete only at the closing fence
        if (text.compare(i, 3, "```") == 0) {
            size_t close = text.find("```", i + 3);
            if (close == std::string::npos) return std::string::npos;
            i = close + 3;
            continue;
        }
        char c = text[i];
        if (c == '`') {   // inline code: skip to its end so a '.' inside is not a boundary
            size_t close = text.find('`', i + 1);
            if (close == std::string::npos) return std::string::npos;
            i = close + 1;
            continue;
        }
        if (c == '\n') {
            // a blank line or a list/heading boundary ends a sentence
            return i + 1;
        }
        if (c == '.' || c == '!' || c == '?') {
            bool next_ws_or_end = (i + 1 >= n) ? false : std::isspace(static_cast<unsigned char>(text[i + 1])) != 0;
            if (i + 1 >= n) return std::string::npos;   // might be "3." of "3.14" still arriving
            bool digit_after = std::isdigit(static_cast<unsigned char>(text[i + 1]));
            bool digit_before = i > from && std::isdigit(static_cast<unsigned char>(text[i - 1]));
            if (c == '.' && (digit_after || (digit_before && !next_ws_or_end))) { ++i; continue; }   // 3.14, 1.2.3
            if (c == '.' && is_abbrev(text, i)) { ++i; continue; }
            if (next_ws_or_end) {
                size_t j = i + 1;
                while (j < n && (text[j] == ')' || text[j] == '"' || text[j] == '\'')) ++j;
                return i + 1;
            }
        }
        ++i;
    }
    return std::string::npos;
}

// ---- arithmetic ----
bool is_op_sym(const std::string& s) { return s == "+" || s == "-" || s == "*" || s == "/" || s == "^"; }

struct OpWord { const char* word; const char* op; };
const OpWord kOpWords[] = {{"plus", "+"}, {"minus", "-"}, {"times", "*"}, {"multiplied", "*"}, {"divided", "/"}, {"over", "/"}};

std::string decimal_to_fraction(const std::string& num) {
    size_t dot = num.find('.');
    if (dot == std::string::npos) return num;
    std::string digits = num.substr(0, dot) + num.substr(dot + 1);
    size_t lead = 0;
    while (lead + 1 < digits.size() && digits[lead] == '0') ++lead;
    digits = digits.substr(lead);
    return "(" + digits + "/1" + std::string(num.size() - dot - 1, '0') + ")";
}

struct ArithmeticClaim {
    std::string expr;          // for the CAS (decimals exact, ops normalised)
    std::string shown;         // for people / normalisation
    std::string value;         // claimed value text
    bool value_is_decimal = false;
    bool approximate = false;
    size_t begin = 0, end = 0;
};

// "<expr> (=|is|equals|gives) <number>" with <expr> built of numbers, operators and parentheses.
std::vector<ArithmeticClaim> find_arithmetic(const std::vector<Tok>& tk) {
    std::vector<ArithmeticClaim> out;
    auto is_link = [&](size_t k) {
        const Tok& t = tk[k];
        if (t.type == T::kSym && t.s == "=") return true;
        if (t.type == T::kWord) {
            std::string w = lower(t.s);
            return w == "is" || w == "equals" || w == "equal" || w == "gives" || w == "makes" || w == "yields" || w == "was";
        }
        return false;
    };
    for (size_t k = 1; k + 1 < tk.size(); ++k) {
        if (!is_link(k)) continue;
        // value after the link (skip "about"/"approximately"/"exactly" and an optional '=' / "to")
        size_t v = k + 1;
        bool approx = false;
        while (v < tk.size() && ((tk[v].type == T::kWord && (lower(tk[v].s) == "about" || lower(tk[v].s) == "approximately" || lower(tk[v].s) == "roughly" || lower(tk[v].s) == "exactly" || lower(tk[v].s) == "to")) ||
                                  (tk[v].type == T::kSym && (tk[v].s == "~=" || tk[v].s == "~")))) {
            if (tk[v].s != "exactly" && tk[v].s != "to") approx = true;
            ++v;
        }
        bool neg = false;
        if (v < tk.size() && tk[v].type == T::kSym && tk[v].s == "-") { neg = true; ++v; }
        if (v >= tk.size() || tk[v].type != T::kNum) continue;
        // the value must end the clause (not "= 5 apples", not "= 5 + 3")
        if (v + 1 < tk.size()) {
            const Tok& nx = tk[v + 1];
            bool ends = nx.type == T::kSym && (nx.s == "," || nx.s == ";" || nx.s == ")" || nx.s == "." || nx.s == "!" || nx.s == "?" || nx.s == ":");
            if (nx.type == T::kWord) { std::string w = lower(nx.s); ends = w == "and" || w == "so" || w == "which" || w == "because" || w == "but" || w == "then" || w == "or" || w == "while"; }
            if (!ends) continue;
        }
        // expression before the link
        int depth = 0;
        size_t b = k;   // exclusive start index walking backwards
        int numbers = 0, ops = 0;
        std::string expr, shown;
        std::vector<std::string> parts;   // reversed
        size_t i = k;
        bool bad = false;
        while (i > 0) {
            const Tok& t = tk[i - 1];
            if (t.type == T::kNum) {
                std::string c = clean_number(t.s);
                parts.push_back(decimal_to_fraction(c)); ++numbers; --i; b = i;
            } else if (t.type == T::kSym && is_op_sym(t.s)) {
                parts.push_back(t.s); ++ops; --i; b = i;
            } else if (t.type == T::kSym && t.s == ")") { parts.push_back(")"); ++depth; --i; b = i; }
            else if (t.type == T::kSym && t.s == "(") { parts.push_back("("); --depth; --i; b = i; }
            else if (t.type == T::kWord) {
                std::string w = lower(t.s);
                bool matched = false;
                for (const auto& ow : kOpWords) {
                    if (w == ow.word) { parts.push_back(ow.op); ++ops; matched = true; break; }
                }
                if (matched) { --i; b = i; if (w == "multiplied" || w == "divided") { /* "multiplied by": the "by" is consumed below */ } }
                else if (w == "by" && i >= 2 && (lower(tk[i - 2].s) == "multiplied" || lower(tk[i - 2].s) == "divided")) { --i; }
                else if (w == "x" && i >= 2 && tk[i - 2].type == T::kNum && i < tk.size() && tk[i].type == T::kNum) { parts.push_back("*"); ++ops; --i; b = i; }
                else if (w == "squared") { parts.push_back("2"); parts.push_back("^"); ++ops; --i; b = i; }
                else break;
            } else break;
            if (depth < 0) { /* unmatched '(' reached the left edge */ }
        }
        if (depth != 0) { bad = true; }
        // a trailing operator or "( )" leftovers: reject
        if (bad || numbers < 2 || ops < 1) continue;
        std::reverse(parts.begin(), parts.end());
        for (const auto& p : parts) { expr += p; shown += p; }
        // an expression that begins with an operator (after trimming "of ...") is not an expression
        if (!parts.empty() && (is_op_sym(parts.front()) && parts.front() != "-")) continue;
        if (!parts.empty() && is_op_sym(parts.back())) continue;
        ArithmeticClaim c;
        c.expr = expr;
        c.shown = shown;
        c.value = (neg ? "-" : "") + clean_number(tk[v].s);
        c.value_is_decimal = tk[v].s.find('.') != std::string::npos && tk[v].s.find(',') == std::string::npos;
        c.approximate = approx;
        c.begin = tk[b].b;
        c.end = tk[v].e;
        out.push_back(std::move(c));
        k = v;
    }
    return out;
}

// ---- key/value ("the port is 8080") ----
struct KvCandidate {
    ClaimKind kind;
    std::string subject, value;
    size_t begin = 0, end = 0;
};

std::string last_subject(const std::vector<Tok>& tk, size_t value_idx) {
    // walk left over stopwords/symbols to the nearest content word(s)
    std::vector<std::string> words;
    size_t i = value_idx;
    while (i > 0 && words.size() < 2) {
        const Tok& t = tk[i - 1];
        if (t.type == T::kWord) {
            std::string w = lower(t.s);
            if (stopwords().count(w)) { if (!words.empty()) break; --i; continue; }
            words.push_back(w);
            --i;
        } else if (t.type == T::kSym && (t.s == "=" || t.s == ":" || t.s == "->")) {
            --i;
        } else if (t.type == T::kTick) {
            words.push_back(lower(strip_ticks(t.s)));
            --i;
        } else break;
    }
    std::reverse(words.begin(), words.end());
    std::string s;
    for (const auto& w : words) s += (s.empty() ? "" : " ") + w;
    return s;
}

bool looks_like_version(const std::string& s) {
    int dots = 0;
    for (char c : s) if (c == '.') ++dots;
    return dots >= 2;
}

bool is_year(const std::string& s) {
    return s.size() == 4 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }) &&
           (s[0] == '1' || s[0] == '2') && s > "1000" && s < "2200";
}

const std::set<std::string>& month_names() {
    static const std::set<std::string> m = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                                            "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"};
    return m;
}

const std::set<std::string>& unit_words() {
    static const std::set<std::string> u = {"gb", "mb", "kb", "tb", "gib", "mib", "kib", "ms", "us", "ns", "s", "sec", "seconds", "minutes", "hours",
                                            "hz", "khz", "mhz", "ghz", "w", "kw", "mw", "tflops", "gflops", "%", "tokens", "bytes", "bits",
                                            "gbps", "mbps", "fps", "cores", "threads", "users", "requests", "layers", "experts"};
    return u;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
struct ClaimDetector::Impl {
    static bool risky(const std::string& subject) {
        std::string last = subject;
        size_t sp = last.rfind(' ');
        if (sp != std::string::npos) last = last.substr(sp + 1);
        return risky_subjects().count(last) > 0;
    }
};

ClaimDetector::ClaimDetector(const DetectorConfig& cfg) : impl_(new Impl()), cfg_(cfg) {}
ClaimDetector::~ClaimDetector() { delete impl_; }

bool ClaimDetector::may_contain_claims(const std::string& text) {
    for (unsigned char c : text) {
        if (std::isdigit(c) || c == '/' || c == '\\' || c == '`' || c == '[' || c == '=' || c == '~') return true;
    }
    // citations without digits ("et al."), and statements about expressions in words
    for (const char* k : {"et al", "http", "derivative", "integral", "expands to", "simplifies to", "factors as", "factors into", "equals", "equal to", "solution"})
        if (text.find(k) != std::string::npos) return true;
    return false;
}

size_t ClaimDetector::complete_sentence_end(const std::string& text, size_t from) {
    size_t pos = from, last = from;
    while (pos < text.size()) {
        size_t e = sentence_end(text, pos);
        if (e == std::string::npos) break;
        last = e;
        pos = e;
    }
    return last;
}

ClaimRisk ClaimDetector::classify(const std::string& text) const {
    if (!may_contain_claims(text)) return ClaimRisk::kLow;
    RequestScope scope;
    auto cs = scan(text, 0, text.size(), scope);
    ClaimRisk r = ClaimRisk::kLow;
    for (const auto& c : cs) if (c.risk > r) r = c.risk;
    return r;
}

std::vector<Claim> ClaimDetector::scan(const std::string& text, size_t from, size_t to, const RequestScope& scope) const {
    std::vector<Claim> claims;
    if (from >= to || to > text.size() || !may_contain_claims(text.substr(from, to - from))) return claims;

    size_t pos = from;
    while (pos < to) {
        size_t e = sentence_end(text, pos);
        if (e == std::string::npos || e > to) e = to;
        if (e <= pos) break;
        const std::string sentence = text.substr(pos, e - pos);
        const size_t base = pos;
        pos = e;
        if (sentence.size() > 6000 || !may_contain_claims(sentence)) continue;

        std::vector<Claim> found;
        auto add = [&](ClaimKind kind, ClaimRisk risk, const std::string& subject, const std::string& value, const std::string& normalized) {
            for (const auto& c : found) if (c.kind == kind && c.normalized == normalized) return;
            if (found.size() >= cfg_.max_claims_per_sentence) return;
            Claim c;
            c.kind = kind;
            c.risk = risk;
            c.text = sentence;
            c.begin = base;
            c.end = base + sentence.size();
            c.subject = subject;
            c.value = value;
            c.normalized = normalized;
            c.id = "c_" + hash128(scope.key() + "|" + std::to_string(c.begin) + "|" + normalized).substr(0, 12);
            found.push_back(std::move(c));
        };

        // fenced code: a code-behaviour claim (verified only where a verifier exists)
        if (sentence.find("```") != std::string::npos) {
            Claim c;
            c.kind = ClaimKind::kCodeBehavior;
            c.risk = ClaimRisk::kHigh;
            c.text = sentence;
            c.begin = base;
            c.end = base + sentence.size();
            c.normalized = "code:" + hash128(sentence).substr(0, 16);
            c.id = "c_" + hash128(scope.key() + "|" + c.normalized).substr(0, 12);
            found.push_back(std::move(c));
        }

        std::vector<Tok> tk = tokenize(sentence);

        // ---- formal: arithmetic ----
        std::vector<ArithmeticClaim> ar = find_arithmetic(tk);
        std::set<size_t> consumed_value_tokens;
        for (const auto& a : ar) {
            if (a.approximate) continue;   // "about" claims are not exact claims
            Claim c;
            c.kind = ClaimKind::kArithmetic;
            c.risk = ClaimRisk::kHigh;
            c.text = sentence;
            c.begin = base;
            c.end = base + sentence.size();
            c.subject = a.shown;
            c.value = a.value;
            c.normalized = "arith:" + a.shown + "=" + a.value;
            c.has_formal = true;
            c.formal.type = logic::ClaimType::kArithmetic;
            c.formal.expression = a.expr;
            c.formal.claimed_value = a.value + (a.value_is_decimal ? "" : "");
            c.formal.raw_statement = sentence;
            c.formal.tenant_id = scope.security.tenant_id;
            c.formal.session_id = scope.security.session_id;
            c.formal.user_id = scope.security.user_id;
            c.formal.metadata["decimal"] = a.value_is_decimal ? "1" : "0";
            c.id = "c_" + hash128(scope.key() + "|" + std::to_string(c.begin) + "|" + c.normalized).substr(0, 12);
            bool dup = false;
            for (const auto& f : found) if (f.normalized == c.normalized) dup = true;
            if (!dup) found.push_back(std::move(c));
        }

        // ---- formal: number theory stated in words ("97 is prime", "the 20th Fibonacci number is 6765") ----
        {
            const std::string ls = lower(sentence);
            if (sentence.size() <= 2000 && (ls.find("prime") != std::string::npos || ls.find("composite") != std::string::npos || ls.find("even") != std::string::npos ||
                ls.find("odd") != std::string::npos || ls.find("fibonacci") != std::string::npos)) {   // std::regex recurses with input length: bounded
                static const std::regex r_prime(R"((\d{1,40})\s+is\s+(not\s+)?(?:a\s+)?(prime|composite)\b)", std::regex::icase);
                static const std::regex r_parity(R"((\d{1,40})\s+is\s+(not\s+)?(?:an?\s+)?(even|odd)\b)", std::regex::icase);
                static const std::regex r_fib(R"((\d{1,6})(?:st|nd|rd|th)\s+fibonacci(?:\s+number)?\s+is\s+(\d{1,80}))", std::regex::icase);
                static const std::regex r_nth_prime(R"((\d{1,7})(?:st|nd|rd|th)\s+prime(?:\s+number)?\s+is\s+(\d{1,12}))", std::regex::icase);
                auto emit = [&](const std::string& shown, const std::string& expr, const std::string& value) {
                    Claim c;
                    c.kind = ClaimKind::kArithmetic;
                    c.risk = ClaimRisk::kHigh;
                    c.text = sentence; c.begin = base; c.end = base + sentence.size();
                    c.subject = shown; c.value = value;
                    c.normalized = "nt:" + expr + "=" + value;
                    c.has_formal = true;
                    c.formal.type = logic::ClaimType::kArithmetic;
                    c.formal.expression = expr;
                    c.formal.claimed_value = value;
                    c.formal.raw_statement = sentence;
                    c.formal.tenant_id = scope.security.tenant_id;
                    c.formal.session_id = scope.security.session_id;
                    c.formal.user_id = scope.security.user_id;
                    c.id = "c_" + hash128(scope.key() + "|" + std::to_string(c.begin) + "|" + c.normalized).substr(0, 12);
                    for (const auto& f : found) if (f.normalized == c.normalized) return;
                    found.push_back(std::move(c));
                };
                std::smatch m;
                for (auto it = std::sregex_iterator(sentence.begin(), sentence.end(), r_prime); it != std::sregex_iterator(); ++it) {
                    const std::string n = (*it)[1], w = lower((*it)[3]);
                    const bool neg = (*it)[2].matched;
                    if (n.size() > 1 && n[0] == '0') continue;
                    if (n == "0" || n == "1") continue;   // neither prime nor composite
                    const bool is_prime_claim = (w == "prime") != neg;
                    emit(n + " is " + (neg ? "not " : "") + w, "primeq(" + n + ")", is_prime_claim ? "True" : "False");
                }
                for (auto it = std::sregex_iterator(sentence.begin(), sentence.end(), r_parity); it != std::sregex_iterator(); ++it) {
                    const std::string n = (*it)[1], w = lower((*it)[3]);
                    const bool neg = (*it)[2].matched;
                    emit(n + " is " + (neg ? "not " : "") + w, std::string(w == "even" ? "evenq(" : "oddq(") + n + ")", neg ? "False" : "True");
                }
                for (auto it = std::sregex_iterator(sentence.begin(), sentence.end(), r_fib); it != std::sregex_iterator(); ++it)
                    emit("Fibonacci(" + std::string((*it)[1]) + ")", "fibonacci(" + std::string((*it)[1]) + ")", (*it)[2]);
                for (auto it = std::sregex_iterator(sentence.begin(), sentence.end(), r_nth_prime); it != std::sregex_iterator(); ++it)
                    emit("prime #" + std::string((*it)[1]), "prime(" + std::string((*it)[1]) + ")", (*it)[2]);
                (void)m;
            }
        }

        // ---- formal: equations and calculus stated in words ("the derivative of x^2 is 2x") ----
        if (sentence.size() <= 1500) {
            for (const auto& sp : find_symbolic_claims(sentence)) {
                Claim c;
                c.kind = ClaimKind::kSymbolic;
                c.risk = ClaimRisk::kHigh;
                c.text = sentence; c.begin = base; c.end = base + sentence.size();
                c.subject = sp.shown; c.value = sp.g;
                c.normalized = "sym:" + sp.kind + ":" + sp.f + "=>" + sp.g + ":" + sp.var;
                c.has_formal = true;
                c.formal.type = logic::ClaimType::kSymbolic;
                c.formal.expression = encode_symbolic(sp);
                c.formal.claimed_value = sp.g;
                c.formal.raw_statement = sentence;
                c.formal.tenant_id = scope.security.tenant_id;
                c.formal.session_id = scope.security.session_id;
                c.formal.user_id = scope.security.user_id;
                c.id = "c_" + hash128(scope.key() + "|" + std::to_string(c.begin) + "|" + c.normalized).substr(0, 12);
                bool dup = false;
                for (const auto& f : found) if (f.normalized == c.normalized) dup = true;
                if (!dup) found.push_back(std::move(c));
            }
        }

        // ---- values attached to a subject ----
        for (size_t i = 0; i < tk.size(); ++i) {
            const Tok& t = tk[i];
            // skip numbers that are part of an arithmetic claim already
            bool in_arith = false;
            for (const auto& a : ar) if (t.b >= a.begin && t.e <= a.end) in_arith = true;
            if (in_arith) continue;

            if (t.type == T::kPath) {
                add(ClaimKind::kFilePath, ClaimRisk::kHigh, "path", t.s, "path:" + t.s);
            } else if (t.type == T::kUrl) {
                add(ClaimKind::kCitation, ClaimRisk::kHigh, "url", t.s, "url:" + lower(t.s));
            } else if (t.type == T::kTick) {
                std::string inner = strip_ticks(t.s);
                if (inner.empty()) continue;
                size_t paren = inner.find('(');
                bool ident_like = std::all_of(inner.begin(), inner.begin() + (paren == std::string::npos ? inner.size() : paren), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '.' || c == ':' || c == '-'; });
                if (!ident_like) continue;
                if (paren != std::string::npos && inner.back() == ')') add(ClaimKind::kApiSignature, ClaimRisk::kHigh, inner.substr(0, paren), inner, "api:" + inner);
                else if (inner.find('/') != std::string::npos || inner.find('\\') != std::string::npos) add(ClaimKind::kFilePath, ClaimRisk::kHigh, "path", inner, "path:" + inner);
                else add(ClaimKind::kIdentifier, ClaimRisk::kMedium, "identifier", inner, "id:" + inner);
            } else if (t.type == T::kNum) {
                std::string raw = clean_number(t.s);
                // date: 2024-03-05, or a year after a month name / in a "in 1999" position
                if (i + 4 < tk.size() + 0 && false) continue;
                bool iso = false;
                if (t.s.size() == 4 && is_year(t.s) && i + 3 < tk.size() + 4) {
                    // YYYY-MM-DD: tokens "2024", "-", "03", "-", "05"
                    if (i + 4 < tk.size() && tk[i + 1].s == "-" && tk[i + 2].type == T::kNum && tk[i + 3].s == "-" && tk[i + 4].type == T::kNum) {
                        std::string d = t.s + "-" + tk[i + 2].s + "-" + tk[i + 4].s;
                        add(ClaimKind::kDate, ClaimRisk::kHigh, "date", d, "date:" + d);
                        i += 4;
                        iso = true;
                    }
                }
                if (iso) continue;
                if (i >= 1 && tk[i - 1].type == T::kWord && month_names().count(lower(tk[i - 1].s)) && raw.size() <= 2) {
                    // "March 5, 2024"
                    std::string d = lower(tk[i - 1].s) + " " + raw;
                    if (i + 2 < tk.size() && tk[i + 1].s == "," && tk[i + 2].type == T::kNum && is_year(tk[i + 2].s)) { d += " " + tk[i + 2].s; i += 2; }
                    add(ClaimKind::kDate, ClaimRisk::kHigh, "date", d, "date:" + d);
                    continue;
                }
                if (i + 1 < tk.size() && tk[i + 1].type == T::kWord && month_names().count(lower(tk[i + 1].s)) && raw.size() <= 2) {
                    std::string d = raw + " " + lower(tk[i + 1].s);
                    size_t used = 1;
                    if (i + 2 < tk.size() && tk[i + 2].type == T::kNum && is_year(tk[i + 2].s)) { d += " " + tk[i + 2].s; used = 2; }
                    add(ClaimKind::kDate, ClaimRisk::kHigh, "date", d, "date:" + d);
                    i += used;
                    continue;
                }
                if (i >= 1 && tk[i - 1].type == T::kWord && month_names().count(lower(tk[i - 1].s)) && is_year(raw)) {
                    add(ClaimKind::kDate, ClaimRisk::kHigh, "date", lower(tk[i - 1].s) + " " + raw, "date:" + lower(tk[i - 1].s) + " " + raw);
                    continue;
                }
                // citation numbers [3]
                if (i >= 1 && i + 1 < tk.size() && tk[i - 1].s == "[" && tk[i + 1].s == "]") {
                    add(ClaimKind::kCitation, ClaimRisk::kHigh, "citation", raw, "cite:" + raw);
                    continue;
                }
                std::string subject = last_subject(tk, i);
                // a unit directly after the number
                std::string unit;
                if (i + 1 < tk.size() && tk[i + 1].type == T::kWord && unit_words().count(lower(tk[i + 1].s))) unit = lower(tk[i + 1].s);
                if (i + 1 < tk.size() && tk[i + 1].s == "%") unit = "%";
                bool version = looks_like_version(t.s);
                if (version) { add(ClaimKind::kConfigValue, ClaimRisk::kHigh, subject.empty() ? "version" : subject, t.s, "kv:" + (subject.empty() ? std::string("version") : subject) + "=" + t.s); continue; }
                // "64 GB of memory" is about memory; "48 layers" is layers = 48
                if (!unit.empty() && i + 3 < tk.size() + 3 && i + 2 < tk.size() && lower(tk[i + 2].s) == "of" && i + 3 < tk.size() && tk[i + 3].type == T::kWord)
                    subject = lower(tk[i + 3].s);
                if (i + 1 < tk.size() && tk[i + 1].type == T::kWord && count_nouns().count(lower(tk[i + 1].s))) { subject = lower(tk[i + 1].s); unit.clear(); }
                if (subject.empty() && unit.empty()) continue;      // a bare number with nothing it is "about"
                if (unusable_subjects().count(subject)) continue;
                // value after a defining verb/symbol, or a risky subject, or a spec phrase
                bool defining = false;
                if (i >= 1) {
                    for (size_t k = i; k > 0 && k + 3 > i; --k) {
                        const Tok& p = tk[k - 1];
                        if (p.type == T::kSym && (p.s == "=" || p.s == ":")) { defining = true; break; }
                        if (p.type == T::kWord) {
                            std::string w = lower(p.s);
                            if (w == "is" || w == "are" || w == "was" || w == "equals" || w == "to" || w == "uses" || w == "use" || w == "has" || w == "have" || w == "of" || w == "set" || w == "on" || w == "at") { defining = true; break; }
                        }
                    }
                }
                bool spec = false;
                for (size_t k = i; k > 0 && k + 4 > i; --k) {
                    const Tok& p = tk[k - 1];
                    if (p.type == T::kWord) {
                        std::string w = lower(p.s);
                        if (w == "supports" || w == "support" || w == "requires" || w == "require" || w == "maximum" || w == "minimum" || w == "max" || w == "min" || w == "default" || w == "defaults" || w == "limit") spec = true;
                    }
                }
                bool risky = Impl::risky(subject);
                if (!(defining || risky || spec || !unit.empty())) continue;
                std::string value = raw + (unit.empty() || unit == "%" ? (unit == "%" ? "%" : "") : " " + unit);
                if (is_year(raw) && subject.empty()) continue;
                ClaimKind kind = spec || !unit.empty() ? ClaimKind::kSpecification : (risky || defining ? ClaimKind::kConfigValue : ClaimKind::kNumber);
                if (kind == ClaimKind::kSpecification && risky) kind = ClaimKind::kConfigValue;
                if (subject.empty()) subject = unit;
                ClaimRisk risk = (kind == ClaimKind::kConfigValue || kind == ClaimKind::kSpecification) ? ClaimRisk::kHigh : ClaimRisk::kMedium;
                add(kind, risk, subject, value, "kv:" + subject + "=" + value);
            } else if (t.type == T::kWord) {
                std::string w = lower(t.s);
                if (w == "rfc" && i + 1 < tk.size() && tk[i + 1].type == T::kNum) { add(ClaimKind::kCitation, ClaimRisk::kHigh, "rfc", tk[i + 1].s, "cite:rfc" + tk[i + 1].s); }
                if (w == "al" && i >= 1 && lower(tk[i - 1].s) == "et") add(ClaimKind::kCitation, ClaimRisk::kHigh, "author", "et al", "cite:etal" + std::to_string(i));
                if ((w == "doi" || w == "arxiv") && i + 1 < tk.size() && tk[i + 1].s == ":") add(ClaimKind::kCitation, ClaimRisk::kHigh, w, w, "cite:" + w + std::to_string(i));
            }
        }

        for (auto& c : found) {
            if (c.risk >= cfg_.min_risk) claims.push_back(std::move(c));
        }
    }
    return claims;
}

} // namespace strata::rt
