package densecpu

// dequantIQ3S expands one 110-byte IQ3_S block: d(2) qs(64) qh(8) signs(32) scales(4), 256 values.
func dequantIQ3S(src []byte, dst []float32) {
	d := f16(src)
	qs, qh, signs, scales := src[2:66], src[66:74], src[74:106], src[106:110]
	for ib := 0; ib < 8; ib++ {
		db := d * float32(1+2*int((scales[ib/2]>>(4*(ib%2)))&0xf))
		for il := 0; il < 4; il++ {
			g1 := iq3sGrid[int(qs[8*ib+2*il])|(int(qh[ib])<<(8-2*il))&256]
			g2 := iq3sGrid[int(qs[8*ib+2*il+1])|(int(qh[ib])<<(7-2*il))&256]
			sg := signs[4*ib+il]
			y := dst[32*ib+8*il:]
			for j := 0; j < 4; j++ {
				a, b := float32(byte(g1>>(8*j))), float32(byte(g2>>(8*j)))
				if sg&(1<<j) != 0 {
					a = -a
				}
				if sg&(1<<(j+4)) != 0 {
					b = -b
				}
				y[j], y[j+4] = db*a, db*b
			}
		}
	}
}

var kvaluesIQ4NL = [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// dequantIQ4XS expands one 136-byte IQ4_XS block: d(2) scales_h(2) scales_l(4) qs(128), 256 values.
func dequantIQ4XS(src []byte, dst []float32) {
	d := f16(src)
	sh := int(src[2]) | int(src[3])<<8
	sl, qs := src[4:8], src[8:136]
	for ib := 0; ib < 8; ib++ {
		ls := int(sl[ib/2]>>(4*(ib%2)))&0xf | (sh>>(2*ib))&3<<4
		dl := d * float32(ls-32)
		for j := 0; j < 16; j++ {
			q := qs[16*ib+j]
			dst[32*ib+j] = dl * kvaluesIQ4NL[q&0xf]
			dst[32*ib+j+16] = dl * kvaluesIQ4NL[q>>4]
		}
	}
}
