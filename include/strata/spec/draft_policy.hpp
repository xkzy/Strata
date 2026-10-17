// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
#pragma once

#include <array>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;
    static constexpr int kAdaptSeen = 8;        // windows of a size before adapt_t trusts its average
    static constexpr int kExploreEvery = 16;    // adapt_t: one round in this many tries the least-seen size

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    /// --spec-auto: the MTP window to verify this round, at most `cap` tokens (1 + drafts). The size whose measured
    /// committed tokens per millisecond is best, once at least kAdaptSeen windows of it have run; a smaller size
    /// replaces the cap only when it wins by `margin`. Every kExploreEvery-th round (`round` counts them) the size seen
    /// least so far is tried instead, so a size that was bad once is measured again as the machine and text change.
    /// Which drafts are verified changes, never the output.
    int adapt_t(int cap, long long round) const;

    /// --lookup-chain: how many of `k_avail` lookup tokens to append after an MTP window of `t_mtp` tokens whose
    /// drafts all hold with probability `p_mtp` (estimated from the draft layer's own probabilities): the k with the
    /// best expected tokens per ms, E(t_mtp) + p_mtp (c + c^2 + .. + c^k) at the cost of t_mtp + k, if it beats the MTP
    /// window alone by `margin`; c = the acceptance of chained drafts by match length (their own counts). 0 = none.
    int chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    /// After a chained round: the MTP window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);
    double chain_rate(int match) const;
    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;

    int max_t_;
    double margin_;
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};      // round ms by window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
};

}  // namespace strata::spec
