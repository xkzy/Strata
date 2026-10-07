// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/discrete.cpp - exact statistics, combinatorics and graph mathematics (clean-room; behaviour follows Mathics / Sage)
//
// Statistics are exact on rational data (sample variance, like Mathics). Combinatorial lists keep Mathematica's order.
// Graphs are given as lists of edges {{1, 2}, {2, 3}, ...}; vertices are numbers or symbols.
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

namespace strata::math::cas {

namespace {

bool numbers_of(const Expr& e, std::vector<Rational>& out) {
    if (!e->has_head("List") || e->args.empty()) return false;
    out.clear();
    for (const auto& a : e->args) { if (!a->is_number()) return false; out.push_back(a->q); }
    return true;
}

bool vertex_less(const Expr& a, const Expr& b) {
    if (a->is_number() && b->is_number()) return a->q < b->q;
    if (a->is_number() != b->is_number()) return a->is_number();   // numbers before symbols
    return to_string(a) < to_string(b);
}

struct Graph {
    std::vector<Expr> vertices;                       // sorted
    std::map<std::string, size_t> index;              // printed vertex -> position
    std::vector<std::vector<size_t>> adj;             // sorted, with multiplicity for multi-edges, self-loops dropped
    size_t edges = 0;
};

bool read_graph(const Expr& e, Graph& g) {
    if (!e->has_head("List")) return false;
    std::vector<std::pair<Expr, Expr>> es;
    for (const auto& edge : e->args) {
        if (!edge->has_head("List", 2)) return false;
        es.push_back({edge->args[0], edge->args[1]});
        if (es.size() > 20000) throw CasLimitError("graph has too many edges");
    }
    std::vector<Expr> vs;
    for (const auto& [a, b] : es) { vs.push_back(a); vs.push_back(b); }
    std::sort(vs.begin(), vs.end(), vertex_less);
    vs.erase(std::unique(vs.begin(), vs.end(), [](const Expr& a, const Expr& b) { return to_string(a) == to_string(b); }), vs.end());
    if (vs.size() > 5000) throw CasLimitError("graph has too many vertices");
    g.vertices = vs;
    for (size_t i = 0; i < vs.size(); ++i) g.index[to_string(vs[i])] = i;
    g.adj.assign(vs.size(), {});
    for (const auto& [a, b] : es) {
        const size_t u = g.index[to_string(a)], v = g.index[to_string(b)];
        if (u == v) continue;
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
        ++g.edges;
    }
    for (auto& n : g.adj) std::sort(n.begin(), n.end());
    return true;
}

std::vector<std::vector<size_t>> components(const Graph& g) {
    std::vector<int> seen(g.vertices.size(), 0);
    std::vector<std::vector<size_t>> out;
    for (size_t s = 0; s < g.vertices.size(); ++s) {
        if (seen[s]) continue;
        std::vector<size_t> comp;
        std::queue<size_t> q;
        q.push(s);
        seen[s] = 1;
        while (!q.empty()) {
            const size_t u = q.front(); q.pop();
            comp.push_back(u);
            for (size_t v : g.adj[u]) if (!seen[v]) { seen[v] = 1; q.push(v); }
        }
        std::sort(comp.begin(), comp.end());
        out.push_back(comp);
    }
    return out;
}

BigInt big(int64_t v) { return BigInt(v); }

} // namespace

Expr Engine::discrete_math(const std::string& head, const std::vector<Expr>& args) {
    std::vector<Rational> xs;

    // ---------------------------------------------------------------- statistics (exact)
    if ((head == "Mean" || head == "Median" || head == "Variance" || head == "StandardDeviation" || head == "Mode") && args.size() == 1) {
        if (!numbers_of(args[0], xs)) return nullptr;
        const size_t n = xs.size();
        if (head == "Mean") { Rational s(0); for (auto& x : xs) s = s + x; return num(s / Rational(static_cast<int64_t>(n))); }
        if (head == "Median") {
            std::sort(xs.begin(), xs.end());
            if (n % 2) return num(xs[n / 2]);
            return num((xs[n / 2 - 1] + xs[n / 2]) / Rational(2));
        }
        if (head == "Mode") {
            std::map<std::string, std::pair<Rational, size_t>> cnt;
            for (auto& x : xs) { auto& c = cnt[to_string(num(x))]; c.first = x; ++c.second; }
            size_t best = 0;
            for (auto& kv : cnt) best = std::max(best, kv.second.second);
            std::vector<Rational> modes;
            for (auto& kv : cnt) if (kv.second.second == best) modes.push_back(kv.second.first);
            std::sort(modes.begin(), modes.end());
            std::vector<Expr> out;
            for (auto& m : modes) out.push_back(num(m));
            return app("List", out);
        }
        if (n < 2) throw CasMathError(head + " needs at least two data points");
        Rational mean(0);
        for (auto& x : xs) mean = mean + x;
        mean = mean / Rational(static_cast<int64_t>(n));
        Rational ss(0);
        for (auto& x : xs) ss = ss + (x - mean) * (x - mean);
        const Rational var = ss / Rational(static_cast<int64_t>(n - 1));   // sample variance, like Mathics
        if (head == "Variance") return num(var);
        return eval(power(num(var), num(Rational(1) / Rational(2))));
    }
    if ((head == "Covariance" || head == "Correlation") && args.size() == 2) {
        std::vector<Rational> ys;
        if (!numbers_of(args[0], xs) || !numbers_of(args[1], ys)) return nullptr;
        if (xs.size() != ys.size() || xs.size() < 2) throw CasMathError(head + " needs two lists of the same length (at least 2)");
        const Rational n(static_cast<int64_t>(xs.size()));
        Rational mx(0), my(0);
        for (auto& x : xs) mx = mx + x;
        for (auto& y : ys) my = my + y;
        mx = mx / n; my = my / n;
        Rational sxy(0), sxx(0), syy(0);
        for (size_t i = 0; i < xs.size(); ++i) { sxy = sxy + (xs[i] - mx) * (ys[i] - my); sxx = sxx + (xs[i] - mx) * (xs[i] - mx); syy = syy + (ys[i] - my) * (ys[i] - my); }
        const Rational d(static_cast<int64_t>(xs.size() - 1));
        if (head == "Covariance") return num(sxy / d);
        if (sxx.is_zero() || syy.is_zero()) throw CasMathError("Correlation is undefined for constant data");
        return eval(times({num(sxy), power(num(sxx * syy), num(Rational(-1) / Rational(2)))}));
    }

    // ---------------------------------------------------------------- combinatorics
    if ((head == "StirlingS2" || head == "StirlingS1") && args.size() == 2 && args[0]->is_integer() && args[1]->is_integer()) {
        int64_t n = 0, k = 0;
        if (!args[0]->q.num.to_int64(n) || !args[1]->q.num.to_int64(k) || n < 0 || k < 0) throw CasMathError(head + " needs non-negative integers");
        if (n > 600) throw CasLimitError(head + " argument too large");
        std::vector<std::vector<BigInt>> t(n + 1, std::vector<BigInt>(n + 2, big(0)));
        t[0][0] = big(1);
        for (int64_t i = 1; i <= n; ++i)
            for (int64_t j = 1; j <= i; ++j) {
                if (head == "StirlingS2") t[i][j] = big(j) * t[i - 1][j] + t[i - 1][j - 1];
                else t[i][j] = t[i - 1][j - 1] - big(i - 1) * t[i - 1][j];   // signed first kind
                tick();
            }
        return num(Rational::from_bigint(k > n ? big(0) : t[n][k]));
    }
    if (head == "BellB" && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 0) throw CasMathError("BellB needs a non-negative integer");
        if (n > 1500) throw CasLimitError("BellB argument too large");
        std::vector<BigInt> row{big(1)};   // Bell triangle
        BigInt bell(1);
        for (int64_t i = 1; i <= n; ++i) {
            std::vector<BigInt> next{row.back()};
            for (size_t j = 0; j < row.size(); ++j) next.push_back(next.back() + row[j]);
            bell = next.front();
            row = std::move(next);
            tick(static_cast<uint64_t>(i));
        }
        return num(Rational::from_bigint(n == 0 ? big(1) : row.front() == row.front() ? bell : bell));
    }
    if (head == "CatalanNumber" && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 0) throw CasMathError("CatalanNumber needs a non-negative integer");
        if (n > 20000) throw CasLimitError("CatalanNumber argument too large");
        BigInt c(1);   // C(n+1) = C(n) * 2(2n+1) / (n+2)
        for (int64_t i = 0; i < n; ++i) { c = c * big(2 * (2 * i + 1)) / big(i + 2); tick(); }
        return num(Rational::from_bigint(c));
    }
    if ((head == "Subfactorial" || head == "Derangements") && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 0) throw CasMathError("Subfactorial needs a non-negative integer");
        if (n > 20000) throw CasLimitError("Subfactorial argument too large");
        BigInt a(1), b(0);   // d0 = 1, d1 = 0, d_n = (n-1)(d_{n-1} + d_{n-2})
        if (n == 0) return integer(1);
        for (int64_t i = 2; i <= n; ++i) { BigInt c = big(i - 1) * (a + b); a = b; b = c; tick(); }
        return num(Rational::from_bigint(b));
    }
    if (head == "IntegerPartitions" && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 0) throw CasMathError("IntegerPartitions needs a non-negative integer");
        if (n > 60) throw CasLimitError("IntegerPartitions: n is limited to 60 (the list grows like exp(sqrt(n)))");
        std::vector<Expr> out;
        std::vector<int64_t> cur;
        std::function<void(int64_t, int64_t)> rec = [&](int64_t left, int64_t maxpart) {
            if (left == 0) { std::vector<Expr> p; for (int64_t v : cur) p.push_back(integer(v)); out.push_back(app("List", p)); tick(); return; }
            for (int64_t part = std::min(left, maxpart); part >= 1; --part) { cur.push_back(part); rec(left - part, part); cur.pop_back(); }
        };
        rec(n, n);
        return app("List", out);
    }
    if (head == "Subsets" && args.size() >= 1 && args[0]->has_head("List")) {
        const size_t n = args[0]->args.size();
        if (n > 16) throw CasLimitError("Subsets: the list is limited to 16 elements");
        int64_t only = -1;
        if (args.size() == 2 && args[1]->has_head("List", 1) && args[1]->args[0]->is_integer()) args[1]->args[0]->q.num.to_int64(only);
        std::vector<Expr> out;
        for (size_t size = 0; size <= n; ++size) {
            if (only >= 0 && static_cast<int64_t>(size) != only) continue;
            std::vector<size_t> idx(size);
            std::function<void(size_t, size_t)> rec = [&](size_t pos, size_t from) {
                if (pos == size) { std::vector<Expr> s; for (size_t i : idx) s.push_back(args[0]->args[i]); out.push_back(app("List", s)); tick(); return; }
                for (size_t i = from; i < n; ++i) { idx[pos] = i; rec(pos + 1, i + 1); }
            };
            rec(0, 0);
        }
        return app("List", out);
    }

    // ---------------------------------------------------------------- graphs (edge lists)
    if ((head == "ConnectedComponents" || head == "IsConnected" || head == "VertexDegrees" || head == "SpanningTreeCount" || head == "ShortestPath" ||
         head == "GraphDistance" || head == "EdgeCount" || head == "VertexCount") && !args.empty() && args[0]->has_head("List")) {
        Graph g;
        if (!read_graph(args[0], g)) throw CasMathError(head + ": the graph must be a list of edges {{a, b}, ...}");
        if (head == "VertexCount") return integer(static_cast<int64_t>(g.vertices.size()));
        if (head == "EdgeCount") return integer(static_cast<int64_t>(g.edges));
        if (head == "VertexDegrees") { std::vector<Expr> d; for (auto& n : g.adj) d.push_back(integer(static_cast<int64_t>(n.size()))); return app("List", d); }
        const auto comps = components(g);
        if (head == "IsConnected") return symbol(comps.size() <= 1 && !g.vertices.empty() ? "True" : "False");
        if (head == "ConnectedComponents") {
            std::vector<std::vector<size_t>> order = comps;
            std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.size() != b.size() ? a.size() > b.size() : a < b; });   // largest first
            std::vector<Expr> out;
            for (auto& c : order) { std::vector<Expr> v; for (size_t i : c) v.push_back(g.vertices[i]); out.push_back(app("List", v)); }
            return app("List", out);
        }
        if (head == "SpanningTreeCount") {
            if (comps.size() != 1) return integer(0);
            const size_t n = g.vertices.size();
            if (n == 1) return integer(1);
            std::vector<std::vector<Expr>> lap(n - 1, std::vector<Expr>(n - 1));   // Kirchhoff: any cofactor of the Laplacian
            for (size_t i = 0; i + 1 < n; ++i)
                for (size_t j = 0; j + 1 < n; ++j) {
                    int64_t v = 0;
                    if (i == j) v = static_cast<int64_t>(g.adj[i].size());
                    else v = -static_cast<int64_t>(std::count(g.adj[i].begin(), g.adj[i].end(), j));
                    lap[i][j] = integer(v);
                }
            std::vector<Expr> rows;
            for (auto& r : lap) rows.push_back(app("List", r));
            return matrix_function("Det", {app("List", rows)});
        }
        // ShortestPath(edges, a, b): breadth-first, smallest-vertex-first among equals; {} when unreachable
        if (args.size() != 3) throw CasMathError(head + " needs the graph and two vertices");
        auto fa = g.index.find(to_string(args[1])), fb = g.index.find(to_string(args[2]));
        if (fa == g.index.end() || fb == g.index.end()) throw CasMathError("a vertex is not in the graph");
        std::vector<int> prev(g.vertices.size(), -2);
        std::queue<size_t> q;
        q.push(fa->second);
        prev[fa->second] = -1;
        while (!q.empty()) {
            const size_t u = q.front(); q.pop();
            for (size_t v : g.adj[u]) if (prev[v] == -2) { prev[v] = static_cast<int>(u); q.push(v); }
        }
        if (prev[fb->second] == -2) return head == "GraphDistance" ? symbol("Infinity") : app("List", {});
        std::vector<size_t> path;
        for (int v = static_cast<int>(fb->second); v != -1; v = prev[v]) path.push_back(static_cast<size_t>(v));
        std::reverse(path.begin(), path.end());
        if (head == "GraphDistance") return integer(static_cast<int64_t>(path.size() - 1));
        std::vector<Expr> out;
        for (size_t v : path) out.push_back(g.vertices[v]);
        return app("List", out);
    }
    return nullptr;
}

} // namespace strata::math::cas
