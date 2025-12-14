---
title: "Homomorphic Encryption Schemes"
date: 2024-12-14
draft: false
category: "crypto"
tags: ["crypto", "homomorphic-encryption", "privacy", "go"]
---

Homomorphic encryption allows computation on encrypted data without decryption, enabling privacy-preserving computations.

## Core Idea

**Homomorphic Encryption (HE)** enables operations on ciphertexts that correspond to operations on plaintexts:

$$\text{Decrypt}(E(m_1) \odot E(m_2)) = m_1 \circ m_2$$

where:
- $E(m)$ is the encryption of message $m$
- $\odot$ is an operation on ciphertexts
- $\circ$ is the corresponding operation on plaintexts

**Types:**
- **Partially Homomorphic (PHE)**: Supports one operation (addition OR multiplication)
- **Somewhat Homomorphic (SHE)**: Supports limited operations
- **Fully Homomorphic (FHE)**: Supports arbitrary computations

---

## Paillier Cryptosystem (Additive Homomorphic)

**Core Idea**: Supports addition of encrypted values and multiplication by plaintext constants.

**Mathematical Foundation**:
- **Key Generation**: 
  - Choose primes $p, q$, compute $n = pq$, $\lambda = \text{lcm}(p-1, q-1)$
  - Public key: $(n, g)$ where $g \in \mathbb{Z}_{n^2}^*$
  - Private key: $\lambda$
- **Encryption**: $E(m) = g^m \cdot r^n \bmod n^2$ where $r \in \mathbb{Z}_n^*$ is random
- **Decryption**: $m = \frac{L(c^\lambda \bmod n^2)}{L(g^\lambda \bmod n^2)} \bmod n$ where $L(x) = \frac{x-1}{n}$
- **Homomorphic Addition**: $E(m_1) \cdot E(m_2) = E(m_1 + m_2)$
- **Homomorphic Scalar Multiplication**: $E(m)^k = E(k \cdot m)$

### Go Implementation

```go
package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

type Paillier struct {
	n, n2, g, lambda *big.Int
}

func NewPaillier(bits int) (*Paillier, error) {
	// Generate primes
	p, err := rand.Prime(rand.Reader, bits/2)
	if err != nil {
		return nil, err
	}
	q, err := rand.Prime(rand.Reader, bits/2)
	if err != nil {
		return nil, err
	}

	// Compute n = p * q
	n := new(big.Int).Mul(p, q)
	n2 := new(big.Int).Mul(n, n)

	// Compute lambda = lcm(p-1, q-1)
	p1 := new(big.Int).Sub(p, big.NewInt(1))
	q1 := new(big.Int).Sub(q, big.NewInt(1))
	lambda := new(big.Int).Div(
		new(big.Int).Mul(p1, q1),
		new(big.Int).GCD(nil, nil, p1, q1),
	)

	// Choose g = n + 1 (simplified)
	g := new(big.Int).Add(n, big.NewInt(1))

	return &Paillier{
		n:      n,
		n2:     n2,
		g:      g,
		lambda: lambda,
	}, nil
}

func (p *Paillier) Encrypt(m *big.Int) (*big.Int, error) {
	// Generate random r
	r, err := rand.Int(rand.Reader, p.n)
	if err != nil {
		return nil, err
	}

	// E(m) = g^m * r^n mod n^2
	gm := new(big.Int).Exp(p.g, m, p.n2)
	rn := new(big.Int).Exp(r, p.n, p.n2)
	
	ciphertext := new(big.Int).Mod(
		new(big.Int).Mul(gm, rn),
		p.n2,
	)

	return ciphertext, nil
}

func (p *Paillier) Decrypt(c *big.Int) (*big.Int, error) {
	// L(c^lambda mod n^2) / L(g^lambda mod n^2) mod n
	clambda := new(big.Int).Exp(c, p.lambda, p.n2)
	glambda := new(big.Int).Exp(p.g, p.lambda, p.n2)

	Lc := new(big.Int).Div(
		new(big.Int).Sub(clambda, big.NewInt(1)),
		p.n,
	)
	Lg := new(big.Int).Div(
		new(big.Int).Sub(glambda, big.NewInt(1)),
		p.n,
	)

	// Compute modular inverse of Lg
	LgInv := new(big.Int).ModInverse(Lg, p.n)
	
	plaintext := new(big.Int).Mod(
		new(big.Int).Mul(Lc, LgInv),
		p.n,
	)

	return plaintext, nil
}

// Homomorphic addition: E(m1) * E(m2) = E(m1 + m2)
func (p *Paillier) Add(c1, c2 *big.Int) *big.Int {
	return new(big.Int).Mod(
		new(big.Int).Mul(c1, c2),
		p.n2,
	)
}

// Homomorphic scalar multiplication: E(m)^k = E(k * m)
func (p *Paillier) ScalarMult(c *big.Int, k *big.Int) *big.Int {
	return new(big.Int).Exp(c, k, p.n2)
}

func main() {
	paillier, err := NewPaillier(512)
	if err != nil {
		panic(err)
	}

	// Encrypt two values
	m1 := big.NewInt(42)
	m2 := big.NewInt(17)

	c1, _ := paillier.Encrypt(m1)
	c2, _ := paillier.Encrypt(m2)

	// Homomorphic addition
	cSum := paillier.Add(c1, c2)
	sum, _ := paillier.Decrypt(cSum)
	fmt.Printf("Encrypted: %d + %d = %d\n", m1.Int64(), m2.Int64(), sum.Int64())

	// Homomorphic scalar multiplication
	cMult := paillier.ScalarMult(c1, big.NewInt(3))
	mult, _ := paillier.Decrypt(cMult)
	fmt.Printf("Encrypted: %d * 3 = %d\n", m1.Int64(), mult.Int64())
}
```

---

## ElGamal Cryptosystem (Multiplicative Homomorphic)

**Core Idea**: Supports multiplication of encrypted values.

**Mathematical Foundation**:
- **Key Generation**: 
  - Choose cyclic group $G$ with generator $g$ and order $q$
  - Private key: $x \in \mathbb{Z}_q$
  - Public key: $h = g^x$
- **Encryption**: $E(m) = (g^r, m \cdot h^r)$ where $r \in \mathbb{Z}_q$ is random
- **Decryption**: $m = \frac{c_2}{c_1^x} = \frac{m \cdot h^r}{g^{rx}} = \frac{m \cdot g^{rx}}{g^{rx}} = m$
- **Homomorphic Multiplication**: $E(m_1) \cdot E(m_2) = (g^{r_1+r_2}, m_1 m_2 \cdot h^{r_1+r_2}) = E(m_1 \cdot m_2)$

### Go Implementation

```go
package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

type ElGamal struct {
	p, g, q *big.Int // p is prime, g is generator, q is order
	x, h    *big.Int // x is private key, h = g^x is public key
}

func NewElGamal(bits int) (*ElGamal, error) {
	// Generate safe prime p = 2q + 1
	var p, q, g *big.Int
	for {
		q, _ = rand.Prime(rand.Reader, bits-1)
		p = new(big.Int).Add(
			new(big.Int).Mul(q, big.NewInt(2)),
			big.NewInt(1),
		)
		if p.ProbablyPrime(20) {
			break
		}
	}

	// Find generator g
	g = big.NewInt(2)
	for {
		if new(big.Int).Exp(g, q, p).Cmp(big.NewInt(1)) == 0 {
			break
		}
		g.Add(g, big.NewInt(1))
	}

	// Generate private key
	x, err := rand.Int(rand.Reader, q)
	if err != nil {
		return nil, err
	}

	// Compute public key
	h := new(big.Int).Exp(g, x, p)

	return &ElGamal{
		p: p,
		g: g,
		q: q,
		x: x,
		h: h,
	}, nil
}

type Ciphertext struct {
	c1, c2 *big.Int
}

func (eg *ElGamal) Encrypt(m *big.Int) (*Ciphertext, error) {
	// Generate random r
	r, err := rand.Int(rand.Reader, eg.q)
	if err != nil {
		return nil, err
	}

	// E(m) = (g^r, m * h^r)
	c1 := new(big.Int).Exp(eg.g, r, eg.p)
	hr := new(big.Int).Exp(eg.h, r, eg.p)
	c2 := new(big.Int).Mod(
		new(big.Int).Mul(m, hr),
		eg.p,
	)

	return &Ciphertext{c1: c1, c2: c2}, nil
}

func (eg *ElGamal) Decrypt(ct *Ciphertext) *big.Int {
	// m = c2 / (c1^x)
	c1x := new(big.Int).Exp(ct.c1, eg.x, eg.p)
	c1xInv := new(big.Int).ModInverse(c1x, eg.p)
	
	m := new(big.Int).Mod(
		new(big.Int).Mul(ct.c2, c1xInv),
		eg.p,
	)

	return m
}

// Homomorphic multiplication: E(m1) * E(m2) = E(m1 * m2)
func (eg *ElGamal) Multiply(ct1, ct2 *Ciphertext) *Ciphertext {
	c1 := new(big.Int).Mod(
		new(big.Int).Mul(ct1.c1, ct2.c1),
		eg.p,
	)
	c2 := new(big.Int).Mod(
		new(big.Int).Mul(ct1.c2, ct2.c2),
		eg.p,
	)
	return &Ciphertext{c1: c1, c2: c2}
}

func main() {
	elgamal, err := NewElGamal(512)
	if err != nil {
		panic(err)
	}

	// Encrypt two values
	m1 := big.NewInt(42)
	m2 := big.NewInt(17)

	ct1, _ := elgamal.Encrypt(m1)
	ct2, _ := elgamal.Encrypt(m2)

	// Homomorphic multiplication
	ctMult := elgamal.Multiply(ct1, ct2)
	mult := elgamal.Decrypt(ctMult)
	fmt.Printf("Encrypted: %d * %d = %d\n", m1.Int64(), m2.Int64(), mult.Int64())
}
```

---

## BGV/BFV (Somewhat Homomorphic)

**Core Idea**: Ring Learning With Errors (RLWE) based scheme supporting addition and multiplication with noise management.

**Mathematical Foundation**:
- Operates on polynomials in ring $R_q = \mathbb{Z}_q[X]/(X^n + 1)$
- **Key Generation**: 
  - Secret key: $s \in R_q$ (small coefficients)
  - Public key: $(a, b = -a \cdot s + e)$ where $a$ is random, $e$ is small error
- **Encryption**: $E(m) = (u, v)$ where $u = a \cdot r + e_1$, $v = b \cdot r + e_2 + m$
- **Decryption**: $m = v + u \cdot s \bmod q$
- **Homomorphic Operations**: 
  - Addition: component-wise addition
  - Multiplication: polynomial multiplication with relinearization

### Go Implementation (Simplified)

```go
package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

// Simplified BGV for demonstration
type BGV struct {
	n int    // polynomial degree
	q *big.Int // modulus
	t *big.Int // plaintext modulus
}

type BGVKey struct {
	sk []*big.Int // secret key (polynomial)
	pk [][]*big.Int // public key
}

type BGVCiphertext struct {
	c0, c1 []*big.Int // polynomial coefficients
}

func NewBGV(n int, qBits int) *BGV {
	q, _ := rand.Prime(rand.Reader, qBits)
	t := big.NewInt(2) // binary plaintext space
	
	return &BGV{
		n: n,
		q: q,
		t: t,
	}
}

func (bgv *BGV) KeyGen() *BGVKey {
	// Generate secret key (small coefficients)
	sk := make([]*big.Int, bgv.n)
	for i := range sk {
		sk[i], _ = rand.Int(rand.Reader, big.NewInt(3))
		sk[i].Sub(sk[i], big.NewInt(1)) // -1, 0, or 1
	}

	// Generate public key
	pk := make([][]*big.Int, 2)
	pk[0] = make([]*big.Int, bgv.n)
	pk[1] = make([]*big.Int, bgv.n)

	// a is random
	for i := range pk[0] {
		pk[0][i], _ = rand.Int(rand.Reader, bgv.q)
	}

	// b = -a * s + e (simplified, no error for demo)
	pk[1] = bgv.polyMul(pk[0], sk)
	bgv.polyNeg(pk[1])

	return &BGVKey{sk: sk, pk: pk}
}

func (bgv *BGV) Encrypt(key *BGVKey, m *big.Int) *BGVCiphertext {
	// Generate random r and errors
	r := make([]*big.Int, bgv.n)
	for i := range r {
		r[i], _ = rand.Int(rand.Reader, big.NewInt(2))
	}

	// u = a * r
	u := bgv.polyMul(key.pk[0], r)
	bgv.polyMod(u, bgv.q)

	// v = b * r + m
	v := bgv.polyMul(key.pk[1], r)
	bgv.polyMod(v, bgv.q)
	
	// Add message (scalar)
	if m.Int64() == 1 {
		v[0].Add(v[0], big.NewInt(1))
	}
	bgv.polyMod(v, bgv.q)

	return &BGVCiphertext{c0: u, c1: v}
}

func (bgv *BGV) Decrypt(key *BGVKey, ct *BGVCiphertext) *big.Int {
	// m = v + u * s mod q mod t
	us := bgv.polyMul(ct.c0, key.sk)
	bgv.polyMod(us, bgv.q)
	
	m := new(big.Int).Add(ct.c1[0], us[0])
	m.Mod(m, bgv.q)
	m.Mod(m, bgv.t)

	return m
}

// Homomorphic addition
func (bgv *BGV) Add(ct1, ct2 *BGVCiphertext) *BGVCiphertext {
	c0 := make([]*big.Int, bgv.n)
	c1 := make([]*big.Int, bgv.n)

	for i := 0; i < bgv.n; i++ {
		c0[i] = new(big.Int).Add(ct1.c0[i], ct2.c0[i])
		c1[i] = new(big.Int).Add(ct1.c1[i], ct2.c1[i])
	}

	bgv.polyMod(c0, bgv.q)
	bgv.polyMod(c1, bgv.q)

	return &BGVCiphertext{c0: c0, c1: c1}
}

// Helper functions
func (bgv *BGV) polyMul(a, b []*big.Int) []*big.Int {
	result := make([]*big.Int, bgv.n)
	for i := range result {
		result[i] = big.NewInt(0)
	}

	for i := 0; i < bgv.n; i++ {
		for j := 0; j < bgv.n; j++ {
			idx := (i + j) % bgv.n
			if i+j >= bgv.n {
				// Handle X^n = -1
				result[idx].Sub(result[idx], new(big.Int).Mul(a[i], b[j]))
			} else {
				result[idx].Add(result[idx], new(big.Int).Mul(a[i], b[j]))
			}
		}
	}

	return result
}

func (bgv *BGV) polyNeg(p []*big.Int) {
	for i := range p {
		p[i].Neg(p[i])
	}
}

func (bgv *BGV) polyMod(p []*big.Int, mod *big.Int) {
	for i := range p {
		p[i].Mod(p[i], mod)
	}
}

func main() {
	bgv := NewBGV(8, 128) // Small parameters for demo
	key := bgv.KeyGen()

	// Encrypt two values
	m1 := big.NewInt(1)
	m2 := big.NewInt(1)

	ct1 := bgv.Encrypt(key, m1)
	ct2 := bgv.Encrypt(key, m2)

	// Homomorphic addition
	ctSum := bgv.Add(ct1, ct2)
	sum := bgv.Decrypt(key, ctSum)
	fmt.Printf("Encrypted: %d + %d = %d\n", m1.Int64(), m2.Int64(), sum.Int64())
}
```

---

## Use Cases

1. **Privacy-Preserving Machine Learning**: Train models on encrypted data
2. **Secure Aggregation**: Compute statistics on encrypted datasets
3. **Private Information Retrieval**: Query databases without revealing queries
4. **Secure Voting**: Tally votes without revealing individual votes
5. **Cloud Computing**: Process encrypted data in untrusted environments

---

## Performance Considerations

- **PHE (Paillier/ElGamal)**: Fast, but limited operations
- **SHE (BGV/BFV)**: Moderate performance, limited depth
- **FHE**: Slow, but supports arbitrary computations

**Trade-offs**:
- Security vs. Performance
- Ciphertext size vs. Computation depth
- Key size vs. Security level

---

