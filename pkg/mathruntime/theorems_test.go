package mathruntime

import (
	"testing"
)

func TestTheoremRegistry_Lookups(t *testing.T) {
	reg := GetTheoremRegistry()
	all := reg.ListAll()
	if len(all) < 5 {
		t.Fatalf("expected at least 5 theorems, got %d", len(all))
	}

	flt := reg.FindTheorem("fermat_little_theorem")
	if flt == nil || flt.Name != "Fermat's Little Theorem" {
		t.Errorf("expected to find Fermat's Little Theorem")
	}

	bayes := reg.FindTheorem("bayes rule")
	if bayes == nil || bayes.ID != "bayes_theorem" {
		t.Errorf("expected to find Bayes' Theorem by alias 'bayes rule'")
	}
}

func TestTheoremRegistry_Verification(t *testing.T) {
	reg := GetTheoremRegistry()

	// Fermat's Little Theorem: 7^(101-1) = 1 mod 101
	res1 := reg.VerifyTheorem("fermat_little_theorem", []string{"7", "101"}, "")
	if !res1.Matches {
		t.Errorf("expected Fermat's Little Theorem to be verified for (7, 101): %s", res1.DiscrepancyDetails)
	}

	// Fermat with composite p=100 -> Mismatch
	res1_comp := reg.VerifyTheorem("fermat_little_theorem", []string{"7", "100"}, "")
	if res1_comp.Matches {
		t.Errorf("expected Fermat's Little Theorem to reject composite modulus 100")
	}

	// Euler Totient: 3^4 = 1 mod 10 (phi(10)=4)
	res2 := reg.VerifyTheorem("euler_totient_theorem", []string{"3", "10"}, "")
	if !res2.Matches {
		t.Errorf("expected Euler Totient to be verified for (3, 10)")
	}

	// Wilson: p=13 -> prime
	res3 := reg.VerifyTheorem("wilson_theorem", []string{"13"}, "")
	if !res3.Matches {
		t.Errorf("expected Wilson's Theorem to verify p=13 is prime")
	}

	// Binomial Theorem: (2 + 3)^4 = 625
	res4 := reg.VerifyTheorem("binomial_theorem", []string{"2", "3", "4"}, "")
	if !res4.Matches {
		t.Errorf("expected Binomial theorem to verify (2+3)^4")
	}

	// Rank-Nullity: rank=2, nullity=1, cols=3
	res5 := reg.VerifyTheorem("rank_nullity_theorem", []string{"2", "1", "3"}, "")
	if !res5.Matches {
		t.Errorf("expected Rank-Nullity (2, 1, 3) to be verified")
	}

	// AM-GM: [2, 4, 8]
	res6 := reg.VerifyTheorem("am_gm_inequality", []string{"2", "4", "8"}, "")
	if !res6.Matches {
		t.Errorf("expected AM-GM [2, 4, 8] to be verified")
	}

	// Bayes: P(A)=0.01, P(B|A)=0.9, P(B|~A)=0.05
	res7 := reg.VerifyTheorem("bayes_theorem", []string{"0.01", "0.9", "0.05"}, "")
	if !res7.Matches {
		t.Errorf("expected Bayes theorem to be verified")
	}

	// Shannon-Hartley: B=1000 Hz, SNR=7 -> C = 1000 * log2(8) = 3000 bits/s
	res8 := reg.VerifyTheorem("shannon_hartley_theorem", []string{"1000", "7"}, "")
	if !res8.Matches {
		t.Errorf("expected Shannon-Hartley theorem to be verified")
	}

	// Lagrange Group Theorem: |G|=24, |H|=6 -> [G:H] = 4
	res9 := reg.VerifyTheorem("lagrange_group_theorem", []string{"24", "6"}, "")
	if !res9.Matches {
		t.Errorf("expected Lagrange group theorem to be verified for (24, 6)")
	}

	// RSA: p=61, q=53 -> N=3233, e=17, d=2753, m=65
	res10 := reg.VerifyTheorem("rsa_correctness_theorem", []string{"65", "17", "2753", "3233"}, "")
	if !res10.Matches {
		t.Errorf("expected RSA correctness to be verified: %s", res10.DiscrepancyDetails)
	}
}
