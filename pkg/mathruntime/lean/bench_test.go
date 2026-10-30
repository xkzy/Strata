package lean

import "testing"

func BenchmarkBuildPrelude(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := BuildPrelude(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProveClaim(b *testing.B) {
	if _, err := Prelude(); err != nil {
		b.Fatal(err)
	}
	for _, claim := range []string{"2^10 = 1024", "123456789 * 987654321 = 121932631112635269", "3 ∣ 12 ∧ 5 < 7 ∧ ¬ (9 = 10)",
		"-3 < 2", "-123456789 * 987654321 = -121932631112635269", "-12 / 7 = -2 ∧ -12 % 7 = 2",
		"-3 ∣ 12 ∧ ¬ (-3 ∣ 13)", "gcd(-6, -9) = 3 ∧ min(-3, 5) = -3",
		"gcd(12, 18) = 6", "gcd(123456789, 987654321) = 9"} {
		b.Run(claim, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := ProveClaim(claim, Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
