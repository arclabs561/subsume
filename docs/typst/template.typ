// Page layout: generous margins for mathematical typesetting (Hardy-style)
#set page(margin: (top: 2.5cm, bottom: 2.5cm, left: 3cm, right: 2.5cm))
#set text(font: ("Linux Libertine", "Times New Roman", "serif"), size: 11pt)
#set heading(numbering: "1.", size: 1.2em, above: 1.2em, below: 0.8em)
#set par(justify: true, leading: 0.75em, first-line-indent: 0pt, spacing: 0.3em)
#set math.equation(numbering: "(1)", number-align: right)
#set enum(indent: 1.5em, body-indent: 0.5em)

// Theorem styling: subtle gray background, elegant border
#let theorem(body) = {
  block(
    fill: rgb("f8f9fa"),
    stroke: 0.5pt + rgb("dee2e6"),
    inset: 1.2em,
    radius: 3pt,
    above: 0.8em,
    below: 0.8em,
    body
  )
}

// Definition styling: lighter background, soft border
#let definition(body) = {
  block(
    fill: rgb("fafbfc"),
    stroke: 0.5pt + rgb("e9ecef"),
    inset: 1em,
    radius: 3pt,
    above: 0.6em,
    below: 0.6em,
    body
  )
}

// Proof styling: minimal, clean background
#let proof(body) = {
  block(
    fill: rgb("fcfcfc"),
    stroke: 0.5pt + rgb("f0f0f0"),
    inset: 1em,
    radius: 3pt,
    above: 0.6em,
    below: 0.6em,
    body
  )
}

// Example styling: warm, inviting background
#let example(body) = {
  block(
    fill: rgb("fffef7"),
    stroke: 0.5pt + rgb("f5e6d3"),
    inset: 1em,
    radius: 3pt,
    above: 0.6em,
    below: 0.6em,
    body
  )
}

