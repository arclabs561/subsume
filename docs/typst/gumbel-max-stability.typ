#import "template.typ": theorem, definition, proof, example

#align(center)[
  #text(24pt, weight: "bold")[Gumbel Max-Stability]
]

#v(1em)

== Motivation

*The "family preservation" puzzle:* When computing box intersections, we need to take maxima and minima of Gumbel-distributed coordinates. A crucial question arises: does the result remain Gumbel-distributed? 

This is like asking: "If I take the maximum of two Gumbel-distributed numbers, do I get another Gumbel-distributed number, or do I get something else entirely?" If the answer were "something else", we'd be in trouble—we'd lose the mathematical structure that makes volume calculations tractable. We'd have to compute volumes for a different distribution, and then another, and another, creating an infinite regress.

Max-stability is the property that answers this question affirmatively. It ensures that the maximum of independent Gumbel random variables is itself Gumbel-distributed (after appropriate normalization). This *algebraic closure* property is what makes Gumbel boxes mathematically elegant: operations on Gumbel boxes produce Gumbel boxes, preserving the family throughout all computations. It's like a mathematical "closed system"—you can perform operations within the system without ever leaving it.

*Historical context:* Max-stability is a fundamental concept in extreme value theory. The Fisher-Tippett-Gnedenko theorem (Fisher & Tippett, 1928; Gnedenko, 1943) is the cornerstone of extreme value theory, analogous to the Central Limit Theorem for sums. The theorem states that if $X_1, ..., X_n$ are independent and identically distributed random variables, and if there exist sequences $a_n > 0$ and $b_n$ such that the normalized maximum $(max(X_1, ..., X_n) - b_n)/a_n$ converges in distribution to a non-degenerate limit $G$, then $G$ must belong to one of three families:

1. *Gumbel (Type I)*: $G(x) = e^(-e^(-x))$ for $x in RR$ (infinite support)
2. *Fréchet (Type II)*: $G(x) = e^(-x^(-alpha))$ for $x > 0$, $alpha > 0$ (bounded below)
3. *Weibull (Type III)*: $G(x) = e^(-(-x)^alpha)$ for $x < 0$, $alpha > 0$ (bounded above)

The Gumbel distribution is the only one of these that is max-stable in the sense that the maximum of independent Gumbel random variables (after appropriate normalization) remains Gumbel-distributed *with the same scale parameter*. This is crucial for box embeddings: Fréchet and Weibull distributions require different normalization schemes that would change the scale parameter, breaking the consistency needed for box operations. The Gumbel distribution's infinite support and consistent scale parameter make it uniquely suited for box embeddings, where we need to preserve the scale parameter $beta$ across all operations.

*Connection to block maxima:* In extreme value theory, the "block maxima" approach considers the maximum value in each block of observations. For example, if we have daily temperature data, we might consider the maximum temperature in each year. The Fisher-Tippett theorem tells us that, under appropriate conditions, these block maxima converge in distribution to one of the three extreme value distributions. The Gumbel distribution (Type I) arises when the underlying distribution has an exponential tail, which includes many common distributions like the normal, exponential, and gamma distributions. This universality property—that many different underlying distributions lead to the same limiting distribution for maxima—is what makes extreme value theory so powerful and why Gumbel distributions are so widely applicable.

== Definition

A distribution $G$ is *max-stable* if, for any $n >= 1$, there exist constants $a_n > 0$ and $b_n$ such that:

$ [G(a_n x + b_n)]^n = G(x) $

Here, $[G(x)]^n$ denotes the $n$-th power of the CDF, which equals the CDF of $max(X_1, ..., X_n)$ where $X_1, ..., X_n$ are independent random variables with distribution $G$. This means the maximum of $n$ independent samples (after appropriate scaling and translation) has the same distribution as a single sample. In other words, the distribution is "closed" under the maximum operation.

== Statement

#theorem[
  *Theorem (Gumbel Max-Stability).* If $G_1, ..., G_k ~ "Gumbel"(mu, beta)$ are independent, then:

  $ max(G_1, ..., G_k) ~ "Gumbel"(mu + beta ln k, beta) $

  The location parameter shifts by $beta ln k$, preserving the Gumbel family.
]

== Proof

The CDF of the maximum of $k$ independent random variables is the product of their CDFs (since the maximum is $<= x$ if and only if all $k$ variables are $<= x$):

$ P(max(G_1, ..., G_k) <= x) = product_(i=1)^k P(G_i <= x) = [G(x)]^k $

For Gumbel distribution $G(x) = e^(-e^(-(x-mu)/beta))$:

$ [G(x)]^k = (e^(-e^(-(x-mu)/beta)))^k = e^(-k e^(-(x-mu)/beta)) $

The CDF of $"Gumbel"(mu + beta ln k, beta)$ is:

$ e^(-e^(-(x-(mu+beta ln k))/beta)) = e^(-e^(-(x-mu)/beta + ln k)) = e^(-k e^(-(x-mu)/beta)) $

This matches exactly, proving max-stability.

== Min-Stability

#theorem[
  *Corollary (Min-Stability).* For MinGumbel, if $G_1, ..., G_k ~ "MinGumbel"(mu, beta)$ are independent:

  $ min(G_1, ..., G_k) ~ "MinGumbel"(mu - beta ln k, beta) $

  The location parameter shifts by $-beta ln k$, preserving the MinGumbel family.
]

*Proof:* The key insight is the relationship between MinGumbel and MaxGumbel via negation. If $G_i ~ "MinGumbel"(mu, beta)$, then $-G_i ~ "MaxGumbel"(-mu, beta)$. 

This follows from the definition: MinGumbel has CDF $F_"Min"(x) = 1 - e^(-e^((x-mu)/beta))$, so:

$ P(-G_i <= x) = P(G_i >= -x) = 1 - F_"Min"(-x) = 1 - (1 - e^(-e^((-x-mu)/beta))) = e^(-e^(-(x+mu)/beta)) $

which is the CDF of $"MaxGumbel"(-mu, beta)$.

Using the identity $min(G_1, ..., G_k) = -max(-G_1, ..., -G_k)$ and applying max-stability to the negated variables:

$ max(-G_1, ..., -G_k) ~ "MaxGumbel"(-mu + beta ln k, beta) $

Therefore:

$ min(G_1, ..., G_k) = -max(-G_1, ..., -G_k) ~ -"MaxGumbel"(-mu + beta ln k, beta) = "MinGumbel"(mu - beta ln k, beta) $

This establishes min-stability: the minimum of independent MinGumbel random variables is itself MinGumbel-distributed, with the location parameter shifted by $-beta ln k$.

== Why It Matters

Max-stability is not merely a theoretical curiosity—it is essential for the computational tractability of Gumbel box embeddings. When we compute box intersections (see the Log-Sum-Exp and Gumbel Intersection document):

- *Intersection minimum*: $z_("cap",i) = max(min_(A,i), min_(B,i))$ where $min_(A,i) ~ "MinGumbel"(mu_(x,i)^A, beta)$ and $min_(B,i) ~ "MinGumbel"(mu_(x,i)^B, beta)$.
  
  By max-stability (applied to the negated MinGumbel variables, which are MaxGumbel), we have:
  $ z_("cap",i) ~ "MaxGumbel"("lse"_beta(mu_(x,i)^A, mu_(x,i)^B), beta) $
  
  where $"lse"_beta$ is the log-sum-exp function (see the Log-Sum-Exp document).

- *Intersection maximum*: $Z_("cap",i) = min(max_(A,i), max_(B,i))$ where $max_(A,i) ~ "MaxGumbel"(mu_(y,i)^A, beta)$ and $max_(B,i) ~ "MaxGumbel"(mu_(y,i)^B, beta)$.
  
  By min-stability (applied to the MaxGumbel variables), we have:
  $ Z_("cap",i) ~ "MinGumbel"("lse"_beta(mu_(y,i)^A, mu_(y,i)^B), beta) $

This *algebraic closure* property means:
1. *Analytical tractability*: We can compute expected volumes using the Bessel function formula (see the Gumbel-Box Volume document) at every step, not just for initial boxes. The formula applies recursively to intersections of intersections.
2. *Consistent structure*: The mathematical framework remains consistent throughout all operations—intersections, unions, and compositions. Every operation produces Gumbel-distributed coordinates.
3. *Computational efficiency*: We avoid the need for numerical integration or Monte Carlo methods at each step. All volume calculations reduce to evaluating the Bessel function $K_0$.

*Why the scale parameter stays the same:* A crucial insight is that max-stability preserves the scale parameter $beta$. When we take the maximum of $k$ Gumbel random variables, only the location parameter shifts (by $beta ln k$), while $beta$ remains unchanged. This ensures that all boxes in a computation share the same scale parameter, maintaining consistency and enabling the use of a single temperature parameter throughout the model.

The logarithmic shift $beta ln k$ appears because the Gumbel CDF has the form $e^(-e^(-(x-mu)/beta))$. When we raise this to the $k$-th power (for the maximum of $k$ independent variables), we get $e^(-k e^(-(x-mu)/beta))$. To match the standard Gumbel form, we need $k e^(-(x-mu)/beta) = e^(-(x-mu-beta ln k)/beta)$, which requires shifting the location parameter by $beta ln k$. This logarithmic dependence on $k$ is characteristic of extreme value distributions and reflects the fact that taking maxima is a multiplicative operation in the exponential scale.

*Intuition for the logarithmic shift:* The shift $beta ln k$ can be understood intuitively: as we take the maximum of more independent Gumbel random variables, the expected value increases. However, the increase is sublinear (logarithmic) rather than linear. This is because extreme values grow slowly: doubling the number of samples doesn't double the expected maximum—it only increases it by $beta ln 2$. This logarithmic growth is a fundamental property of extreme value distributions and is why the Gumbel distribution is sometimes called the "double exponential" distribution: it involves exponentials of exponentials, leading to this logarithmic scaling behavior.

*Quantitative example:* For $G_1, ..., G_k ~ "Gumbel"(0, 1)$, the expected maximum is $E[max(G_1, ..., G_k)] = gamma + ln k$, where $gamma approx 0.5772$ is Euler's constant. For $k = 1$, the expected value is $gamma approx 0.5772$. For $k = 10$, it's $gamma + ln 10 approx 2.88$. For $k = 100$, it's $gamma + ln 100 approx 5.18$. Notice that increasing $k$ by a factor of 10 increases the expected maximum by approximately $ln 10 approx 2.3$, demonstrating the logarithmic growth. This slow growth is why max-stability is computationally efficient: even when taking maxima of many variables, the location parameter grows only logarithmically, keeping the distribution well-behaved.

Without max-stability, each intersection would produce a different distribution family, making analytical calculations impossible and forcing us to resort to expensive numerical methods.

== Example

#example[
  Three independent Gumbel random variables:
  - $G_1, G_2, G_3 ~ "Gumbel"(0, 1)$

  *Visual representation:* Max-stability ensures that taking the maximum preserves the Gumbel family. The flow diagram below illustrates how three independent Gumbel random variables combine to produce another Gumbel-distributed variable:

  #align(center)[
    #block(
      width: 100%,
      inset: 1em,
      fill: rgb("fafafa"),
      radius: 4pt,
      [
        #grid(
          columns: 3,
          column-gutter: 1em,
          row-gutter: 0.5em,
          [
            #block(
              fill: rgb("f0f8ff"),
              stroke: 1pt + rgb("2c3e50"),
              radius: 2pt,
              inset: 0.4em,
              align(center)[$G_1$]
            ),
            #block(
              fill: rgb("f0f8ff"),
              stroke: 1pt + rgb("2c3e50"),
              radius: 2pt,
              inset: 0.4em,
              align(center)[$G_2$]
            ),
            #block(
              fill: rgb("f0f8ff"),
              stroke: 1pt + rgb("2c3e50"),
              radius: 2pt,
              inset: 0.4em,
              align(center)[$G_3$]
            ),
            #block(
              width: 100%,
              align(center)[
                #text(10pt)[$↓$]
                #v(0.2em)
                #text(9pt)[$max$]
              ]
            ),
            #block(
              width: 100%,
              fill: rgb("e0f0ff"),
              stroke: 1.5pt + rgb("2c3e50"),
              radius: 2pt,
              inset: 0.5em,
              align(center)[$max(G_1, G_2, G_3)$]
            ),
            #block(
              width: 100%,
              align(center)[
                #text(10pt)[$↓$]
                #v(0.2em)
                #text(9pt)[$~$]
              ]
            ),
            #block(
              width: 100%,
              fill: rgb("d0e8f0"),
              stroke: 1.5pt + rgb("27ae60"),
              radius: 2pt,
              inset: 0.5em,
              align(center)[$"Gumbel"(ln 3, 1)$]
            ),
          ]
        )
      ]
    )
  ]

  *Max-stability property:*
  $ max(G_1, G_2, G_3) ~ "Gumbel"(ln 3, 1) = "Gumbel"(1.099, 1) $

  The location parameter shifts by $beta ln 3 = ln 3 approx 1.099$, while the scale parameter $beta = 1$ remains unchanged. This demonstrates that the Gumbel family is closed under the maximum operation.

  *Verification:* The CDF of the maximum is:
  $ P(max(G_1, G_2, G_3) <= x) = [e^(-e^(-x))]^3 = e^(-3 e^(-x)) = e^(-e^(-(x-ln 3))) $

  which is the CDF of $"Gumbel"(ln 3, 1)$.
]

== Notes

*Why only Gumbel?* The Fisher-Tippett-Gnedenko theorem identifies three possible limiting distributions for normalized maxima: Gumbel (Type I), Fréchet (Type II), and Weibull (Type III). However, only the Gumbel distribution is max-stable in the sense that the maximum of independent Gumbel random variables remains Gumbel-distributed *with the same scale parameter* (after appropriate location shift). 

Fréchet distributions are max-stable, but the normalization requires scaling by $n^(1/alpha)$ where $alpha$ is the shape parameter. This means the scale parameter changes with the number of variables, breaking consistency. Weibull distributions similarly require normalization that changes the scale parameter. 

For box embeddings, we need the scale parameter $beta$ to remain constant across all operations—when we compute intersections of intersections, we must preserve $beta$ to maintain the mathematical structure. The Gumbel distribution is unique in providing max-stability with scale preservation: $max(G_1, ..., G_k) ~ "Gumbel"(mu + beta ln k, beta)$, where only the location parameter shifts while $beta$ remains unchanged. This algebraic closure property is essential for the computational tractability of box embeddings.

*Connection to extreme value theory:* Max-stability is the defining property of extreme value distributions. In extreme value theory, we're interested in the distribution of maxima (or minima) of large samples. The Fisher-Tippett theorem tells us that, under appropriate conditions, these maxima converge to one of three distributions. The Gumbel distribution (Type I) arises when the underlying distribution has an exponential tail, which includes many common distributions (normal, exponential, gamma). This universality property—that many different underlying distributions lead to the same limiting distribution—is what makes extreme value theory so powerful and why Gumbel distributions are so widely applicable.

*Beyond box embeddings:* Max-stability appears in many contexts beyond box embeddings: in reliability theory (modeling system failures), in finance (modeling extreme market events), and in machine learning (the Gumbel-Max trick for categorical sampling). The property that "the maximum of maxima is still a maximum" (in distribution) is fundamental to understanding extreme events and makes Gumbel distributions the natural choice for modeling extremes.

