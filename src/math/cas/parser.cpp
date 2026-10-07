// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/parser.cpp - recursive-descent parser for the native CAS
#include "strata/math/cas/parser.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>

namespace strata::math::cas {

namespace {

// Known functions: lower-case spelling -> internal (Wolfram-style) head.
const std::map<std::string, std::string>& function_names() {
    static const std::map<std::string, std::string> m = {
        {"sin", "Sin"}, {"cos", "Cos"}, {"tan", "Tan"}, {"cot", "Cot"}, {"sec", "Sec"}, {"csc", "Csc"},
        {"asin", "ArcSin"}, {"acos", "ArcCos"}, {"atan", "ArcTan"}, {"arcsin", "ArcSin"}, {"arccos", "ArcCos"},
        {"arctan", "ArcTan"}, {"sinh", "Sinh"}, {"cosh", "Cosh"}, {"tanh", "Tanh"},
        {"asinh", "ArcSinh"}, {"acosh", "ArcCosh"}, {"atanh", "ArcTanh"},
        {"exp", "Exp"}, {"log", "Log"}, {"ln", "Log"}, {"sqrt", "Sqrt"}, {"abs", "Abs"}, {"sign", "Sign"},
        {"binomial", "Binomial"}, {"factorial", "Factorial"}, {"det", "Det"}, {"inverse", "Inverse"},
        {"transpose", "Transpose"}, {"dot", "Dot"}, {"n", "N"}, {"d", "D"},
        {"floor", "Floor"}, {"ceiling", "Ceiling"}, {"gamma", "Gamma"}, {"min", "Min"}, {"max", "Max"},
        {"gcd", "GCD"}, {"lcm", "LCM"}, {"mod", "Mod"}, {"expand", "Expand"}, {"factor", "Factor"},
        {"simplify", "Simplify"}, {"integrate", "Integrate"}, {"integral", "Integrate"}, {"limit", "Limit"},
        {"solve", "Solve"}, {"roots", "Solve"}, {"diff", "D"}, {"derivative", "D"}, {"taylor", "Series"},
        {"permutations", "Permutations"}, {"multinomial", "Multinomial"}, {"power", "Power"},
        {"plus", "Plus"}, {"times", "Times"}, {"list", "List"}, {"matrix", "Matrix"}, {"series", "Series"},
        {"together", "Together"}, {"cancel", "Cancel"},
        {"primeq", "PrimeQ"}, {"isprime", "PrimeQ"}, {"is_prime", "PrimeQ"}, {"is_pseudoprime", "PrimeQ"},
        {"prime", "Prime"}, {"primepi", "PrimePi"}, {"nextprime", "NextPrime"}, {"next_prime", "NextPrime"},
        {"previous_prime", "PreviousPrime"}, {"prime_range", "Range"}, {"primes", "Range"},
        {"factorinteger", "FactorInteger"}, {"factorint", "FactorInteger"}, {"divisors", "Divisors"},
        {"eulerphi", "EulerPhi"}, {"euler_phi", "EulerPhi"}, {"phi", "EulerPhi"}, {"totient", "EulerPhi"},
        {"moebiusmu", "MoebiusMu"}, {"moebius", "MoebiusMu"}, {"divisorsigma", "DivisorSigma"},
        {"sigma", "DivisorSigma"}, {"powermod", "PowerMod"}, {"power_mod", "PowerMod"},
        {"extendedgcd", "ExtendedGCD"}, {"xgcd", "ExtendedGCD"}, {"jacobisymbol", "JacobiSymbol"},
        {"jacobi_symbol", "JacobiSymbol"}, {"kronecker", "JacobiSymbol"}, {"fibonacci", "Fibonacci"},
        {"fib", "Fibonacci"}, {"lucas_number", "LucasL"}, {"lucasl", "LucasL"},
        {"integerdigits", "IntegerDigits"}, {"integerlength", "IntegerLength"}, {"fromdigits", "FromDigits"},
        {"quotient", "Quotient"}, {"coprimeq", "CoprimeQ"}, {"divisible", "Divisible"}, {"evenq", "EvenQ"},
        {"oddq", "OddQ"}, {"integerq", "IntegerQ"}, {"range", "Range"}, {"table", "Table"}, {"sum", "Sum"},
        {"product", "Product"}, {"total", "Total"}, {"length", "Length"},
        {"apart", "Apart"}, {"collect", "Collect"}, {"rank", "Rank"}, {"matrixrank", "Rank"}, {"tr", "Tr"}, {"trace", "Tr"},
        {"nullspace", "NullSpace"}, {"kernel", "NullSpace"}, {"characteristicpolynomial", "CharacteristicPolynomial"}, {"charpoly", "CharacteristicPolynomial"},
        {"eigenvalues", "Eigenvalues"}, {"eigenvectors", "Eigenvectors"}, {"lu", "LU"}, {"identitymatrix", "IdentityMatrix"},
        {"grad", "Grad"}, {"gradient", "Grad"}, {"jacobian", "Jacobian"}, {"hessian", "Hessian"}, {"reduce", "Reduce"},
        {"chineseremainder", "ChineseRemainder"}, {"crt", "ChineseRemainder"}, {"modularinverse", "ModularInverse"}, {"modinv", "ModularInverse"},
        {"inverse_mod", "ModularInverse"}, {"partitionsp", "PartitionsP"}, {"number_of_partitions", "PartitionsP"},
        {"less", "Less"}, {"greater", "Greater"}, {"lessequal", "LessEqual"}, {"greaterequal", "GreaterEqual"}, {"unequal", "Unequal"},
        {"and", "And"}, {"or", "Or"}, {"not", "Not"},
        {"polynomialmod", "PolynomialMod"}, {"factormod", "FactorMod"}, {"polynomialgcdmod", "PolynomialGCDMod"}, {"multiplicativeorder", "MultiplicativeOrder"},
        {"multiplicative_order", "MultiplicativeOrder"}, {"primitiveroot", "PrimitiveRoot"}, {"primitive_root", "PrimitiveRoot"},
    };
    return m;
}

enum class Tok { End, Num, Ident, Op, LParen, RParen, LBrack, RBrack, LBrace, RBrace, Comma };

struct Token {
    Tok t = Tok::End;
    std::string s;
    size_t pos = 0;
};

class Lexer {
public:
    explicit Lexer(const std::string& text) : s_(text) {}
    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            skip_ws();
            Token tk;
            tk.pos = i_;
            if (i_ >= s_.size()) { tk.t = Tok::End; out.push_back(tk); return out; }
            char c = s_[i_];
            if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i_ + 1 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_ + 1])))) {
                size_t j = i_;
                while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j;
                if (j < s_.size() && s_[j] == '.') {
                    ++j;
                    while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j;
                }
                if (j < s_.size() && (s_[j] == 'e' || s_[j] == 'E') && j + 1 < s_.size() &&
                    (std::isdigit(static_cast<unsigned char>(s_[j + 1])) ||
                     ((s_[j + 1] == '-' || s_[j + 1] == '+') && j + 2 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j + 2]))))) {
                    j += 2;
                    while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j;
                }
                tk.t = Tok::Num; tk.s = s_.substr(i_, j - i_); i_ = j;
            } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                size_t j = i_;
                while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) || s_[j] == '_')) ++j;
                tk.t = Tok::Ident; tk.s = s_.substr(i_, j - i_); i_ = j;
            } else if (c == '(') { tk.t = Tok::LParen; ++i_; }
            else if (c == ')') { tk.t = Tok::RParen; ++i_; }
            else if (c == '[') { tk.t = Tok::LBrack; ++i_; }
            else if (c == ']') { tk.t = Tok::RBrack; ++i_; }
            else if (c == '{') { tk.t = Tok::LBrace; ++i_; }
            else if (c == '}') { tk.t = Tok::RBrace; ++i_; }
            else if (c == ',') { tk.t = Tok::Comma; ++i_; }
            else {
                tk.t = Tok::Op;
                static const char* two[] = {"==", "->", "**", "<=", ">=", "!="};
                bool matched = false;
                for (const char* o : two) {
                    if (s_.compare(i_, 2, o) == 0) { tk.s = o; i_ += 2; matched = true; break; }
                }
                if (!matched) {
                    if (std::string("+-*/^!=<>").find(c) == std::string::npos)
                        throw CasParseError(std::string("unexpected character '") + c + "'");
                    tk.s = std::string(1, c); ++i_;
                }
            }
            out.push_back(tk);
        }
    }
private:
    void skip_ws() { while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_; }
    const std::string& s_;
    size_t i_ = 0;
};

class Parser {
public:
    Parser(std::vector<Token> toks, size_t max_depth) : t_(std::move(toks)), max_depth_(max_depth) {}

    Expr parse_all() {
        Expr e = parse_equation();
        if (cur().t != Tok::End) throw CasParseError("unexpected token '" + cur().s + "'");
        return e;
    }

private:
    std::vector<Token> t_;
    size_t i_ = 0;
    size_t depth_ = 0;
    size_t max_depth_;

    const Token& cur() const { return t_[i_]; }
    const Token& peek(size_t k = 1) const { return t_[std::min(i_ + k, t_.size() - 1)]; }
    bool is_op(const char* s) const { return cur().t == Tok::Op && cur().s == s; }

    struct DepthGuard {
        Parser& p;
        explicit DepthGuard(Parser& pp) : p(pp) { if (++p.depth_ > p.max_depth_) throw CasLimitError("expression nesting too deep"); }
        ~DepthGuard() { --p.depth_; }
    };

    Expr parse_equation() {
        Expr lhs = parse_sum();
        if (is_op("==")) { ++i_; return apply2("Equal", lhs, parse_sum()); }
        if (is_op("<")) { ++i_; return apply2("Less", lhs, parse_sum()); }
        if (is_op(">")) { ++i_; return apply2("Greater", lhs, parse_sum()); }
        if (is_op("<=")) { ++i_; return apply2("LessEqual", lhs, parse_sum()); }
        if (is_op(">=")) { ++i_; return apply2("GreaterEqual", lhs, parse_sum()); }
        if (is_op("!=")) { ++i_; return apply2("Unequal", lhs, parse_sum()); }
        if (is_op("->")) { ++i_; return apply2("Rule", lhs, parse_sum()); }
        return lhs;
    }

    Expr parse_sum() {
        DepthGuard g(*this);
        std::vector<Expr> terms;
        bool neg = false;
        if (is_op("+")) ++i_;
        else if (is_op("-")) { neg = true; ++i_; }
        Expr first = parse_product();
        terms.push_back(neg ? apply2("Times", minus_one(), first) : first);
        while (is_op("+") || is_op("-")) {
            bool minus = cur().s == "-";
            ++i_;
            Expr t = parse_product();
            terms.push_back(minus ? apply2("Times", minus_one(), t) : t);
        }
        return terms.size() == 1 ? terms[0] : app("Plus", std::move(terms));
    }

    bool starts_factor() const {
        const Token& k = cur();
        return k.t == Tok::Num || k.t == Tok::Ident || k.t == Tok::LParen || k.t == Tok::LBrace;
    }

    Expr parse_product() {
        // one flat Times(a, b, c, ...): a left-nested chain of 300,000 factors would be a 300,000-deep tree that
        // overflows the stack when it is evaluated or destroyed
        std::vector<Expr> factors{parse_unary()};
        while (true) {
            if (is_op("*")) { ++i_; factors.push_back(parse_unary()); }
            else if (is_op("/")) { ++i_; factors.push_back(apply2("Power", parse_unary(), minus_one())); }
            else if (starts_factor()) { factors.push_back(parse_unary()); }  // implicit multiplication
            else break;
            if (factors.size() > kMaxFlatTerms) throw CasLimitError("expression has too many factors");
        }
        return factors.size() == 1 ? factors[0] : app("Times", std::move(factors));
    }

    static constexpr size_t kMaxFlatTerms = 100000;

    Expr parse_unary() {
        DepthGuard g(*this);   // "- - - ... x" recurses once per sign
        if (is_op("-")) { ++i_; return apply2("Times", minus_one(), parse_unary()); }
        if (is_op("+")) { ++i_; return parse_unary(); }
        return parse_power();
    }

    Expr parse_power() {
        DepthGuard g(*this);
        Expr base = parse_postfix();
        if (is_op("^") || is_op("**")) {
            ++i_;
            Expr exp = parse_unary();  // right associative; allows 2^-1
            return apply2("Power", base, exp);
        }
        return base;
    }

    Expr parse_postfix() {
        Expr e = parse_atom();
        size_t bangs = 0;
        while (is_op("!") && !(peek().t == Tok::Op && peek().s == "=")) {
            if (++bangs > max_depth_) throw CasLimitError("expression nesting too deep");   // x!!!!... nests one level per '!'
            ++i_;
            e = apply1("Factorial", e);
        }
        return e;
    }

    std::vector<Expr> parse_args(Tok close) {
        std::vector<Expr> args;
        if (cur().t == close) { ++i_; return args; }
        while (true) {
            args.push_back(parse_equation());
            if (cur().t == Tok::Comma) { ++i_; continue; }
            if (cur().t == close) { ++i_; break; }
            throw CasParseError("expected ',' or closing bracket");
        }
        return args;
    }

    Expr parse_atom() {
        DepthGuard g(*this);
        const Token tk = cur();
        switch (tk.t) {
            case Tok::Num: {
                ++i_;
                if (tk.s.find_first_of(".eE") == std::string::npos) {
                    BigInt v;
                    if (!BigInt::from_string(tk.s, v)) throw CasParseError("bad number '" + tk.s + "'");
                    return num(Rational::from_bigint(v));
                }
                return real(std::strtod(tk.s.c_str(), nullptr));
            }
            case Tok::LParen: {
                ++i_;
                Expr e = parse_equation();
                if (cur().t != Tok::RParen) throw CasParseError("expected ')'");
                ++i_;
                return e;
            }
            case Tok::LBrace: {
                ++i_;
                return app("List", parse_args(Tok::RBrace));
            }
            case Tok::LBrack: {
                ++i_;
                return app("List", parse_args(Tok::RBrack));
            }
            case Tok::Ident: {
                ++i_;
                std::string lower = tk.s;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
                auto it = function_names().find(lower);
                if (cur().t == Tok::LBrack) {
                    // f[x, y]: always a call; known names normalise to the internal head.
                    ++i_;
                    std::string head = it != function_names().end() ? it->second : tk.s;
                    auto args = parse_args(Tok::RBrack);
                    if (head == "Matrix" && args.size() == 1 && args[0]->has_head("List")) {
                        return args[0];
                    }
                    return app(head, std::move(args));
                }
                // Single-letter names (n, d, ...) are variables unless written with brackets: n(x+1) is a product.
                if (it != function_names().end() && cur().t == Tok::LParen && tk.s.size() > 1) {
                    ++i_;
                    std::string head = it->second;
                    auto args = parse_args(Tok::RParen);
                    if (head == "Matrix" && args.size() == 1 && args[0]->has_head("List")) {
                        return args[0];
                    }
                    return app(head, std::move(args));
                }
                if (lower == "pi") return symbol("Pi");
                if (lower == "infinity" || lower == "inf" || lower == "oo") return symbol("Infinity");
                if (tk.s == "E") return symbol("E");
                if (tk.s == "I") return symbol("I");
                return symbol(tk.s);
            }
            default:
                throw CasParseError(tk.t == Tok::End ? std::string("unexpected end of expression") : "unexpected token '" + tk.s + "'");
        }
    }
};

} // namespace

Expr parse(const std::string& text, size_t max_depth) {
    if (text.size() > (1u << 20)) throw CasLimitError("expression is too long");
    Lexer lx(text);
    Parser p(lx.run(), max_depth);
    return p.parse_all();
}

} // namespace strata::math::cas
