// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/matrix_util.hpp - list-of-lists matrix helpers shared by the CAS algorithm files (internal)
#pragma once

#include "strata/math/cas/engine.hpp"

namespace strata::math::cas {

using Matrix = std::vector<std::vector<Expr>>;

inline bool as_matrix(const Expr& e, Matrix& m) {
    if (!e->has_head("List") || e->args.empty()) return false;
    m.clear();
    size_t cols = 0;
    for (const auto& row : e->args) {
        if (!row->has_head("List") || row->args.empty()) return false;
        if (cols == 0) cols = row->args.size();
        if (row->args.size() != cols) throw CasMathError("matrix rows have different lengths");
        m.push_back(row->args);
    }
    return true;
}

inline Expr from_matrix(const Matrix& m) {
    std::vector<Expr> rows;
    for (const auto& r : m) rows.push_back(app("List", r));
    return app("List", rows);
}

inline bool all_numbers(const Matrix& m) {
    for (const auto& r : m) for (const auto& e : r) if (!e->is_number()) return false;
    return true;
}

} // namespace strata::math::cas
