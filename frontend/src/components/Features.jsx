
const FEATURES = [
  {
    number: "01",
    title: "Clinical summarization",
    description:
      "Create concise summaries from lengthy clinical notes while retaining the context of the original text.",
    icon: "✳",
  },
  {
    number: "02",
    title: "Information verification",
    description:
      "Check configured clinical terms against the source note and generated summary.",
    icon: "⌕",
  },
  {
    number: "03",
    title: "Omission screening",
    description:
      "Highlight possible missing information for a human reviewer to investigate.",
    icon: "↗",
  },
];

function Features() {
  return (
    <div className="feature-grid">
      {FEATURES.map((feature) => (
        <article className="feature-card" key={feature.number}>
          <div className="feature-card-top">
            <span className="feature-icon" aria-hidden="true">
              {feature.icon}
            </span>

            <span className="feature-number">{feature.number}</span>
          </div>

          <h3>{feature.title}</h3>

          <p>{feature.description}</p>

          <a href="#workspace" className="feature-link">
            Explore workspace <span aria-hidden="true">↗</span>
          </a>
        </article>
      ))}
    </div>
  );
}

export default Features;
