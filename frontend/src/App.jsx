import { useState } from "react";
import "./App.css";
import Features from "./components/Features.jsx";
import NoteInput from "./components/NoteInput.jsx";
import Results from "./components/Results.jsx";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [clinicalNote, setClinicalNote] = useState("");
  const [numSentences, setNumSentences] = useState(8);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSummarize() {
    if (!clinicalNote.trim()) {
      setError("Please enter a clinical note first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          clinical_note: clinicalNote,
          num_sentences: Number(numSentences),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The request could not be completed."
        );
      }

      setResult(data);

      window.setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setClinicalNote("");
    setResult(null);
    setError("");
  }

  function scrollToWorkspace() {
    document.getElementById("workspace")?.scrollIntoView({
      behavior: "smooth",
    });
  }

  return (
    <div className="app-shell" id="home">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="SafeDischargeAI home">
          <span className="brand-icon">+</span>
          <span>
            SafeDischarge<span className="brand-ai">AI</span>
          </span>
        </a>

        <nav className="navigation" aria-label="Main navigation">
          <a href="#features">Features</a>
          <a href="#workspace">Workspace</a>
          <a href="#about">About</a>
        </nav>

        <div className="header-right">
          <span className="prototype-tag">
            <span className="status-dot" />
            RESEARCH PROTOTYPE
          </span>

          <button className="nav-cta" onClick={scrollToWorkspace}>
            Open workspace <span aria-hidden="true">↗</span>
          </button>
        </div>
      </header>

      <main>
        {/* HERO SECTION */}
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              CLINICAL NLP · RESEARCH PROJECT
            </div>

            <h1>
              Less noise.
              <br />
              <span>More clarity</span> in
              <br />
              clinical notes.
            </h1>

            <p className="hero-description">
              Turn lengthy clinical text into concise summaries and screen for
              information that may need a closer look. Designed to support
              review, not replace clinical judgment.
            </p>

            <div className="hero-actions">
              <button
                className="primary-button hero-button"
                onClick={scrollToWorkspace}
              >
                Try the workspace <span aria-hidden="true">→</span>
              </button>

              <a className="text-link" href="#features">
                Explore features <span aria-hidden="true">↓</span>
              </a>
            </div>

            <div className="hero-footnote">
              <span className="footnote-icon">i</span>
              Use synthetic or de-identified demo text only. Do not enter
              identifiable patient information.
            </div>
          </div>

          <div
            className="hero-visual"
            aria-label="Illustration of a clinical note review dashboard"
          >
            <div className="visual-orbit orbit-one" />
            <div className="visual-orbit orbit-two" />

            <div className="visual-card">
              <div className="visual-card-top">
                <span className="mini-brand">
                  <span className="mini-brand-icon">+</span>
                  Clinical review
                </span>

                <span className="demo-pill">DEMO</span>
              </div>

              <div className="document-preview">
                <div className="document-label">SOURCE NOTE</div>
                <div className="skeleton-line long" />
                <div className="skeleton-line medium" />
                <div className="skeleton-line long" />
                <div className="skeleton-line short" />
              </div>

              <div className="visual-arrow">↓</div>

              <div className="summary-preview">
                <div className="preview-heading">
                  <span className="preview-check">✓</span>
                  Summary generated
                </div>

                <div className="skeleton-line long" />
                <div className="skeleton-line medium" />
              </div>

              <div className="review-preview">
                <span className="review-dot" />

                <div>
                  <strong>Review flags</strong>
                  <p>Possible omissions to inspect</p>
                </div>

                <span className="review-arrow">↗</span>
              </div>
            </div>

            <div className="floating-note floating-note-top">
              <span className="floating-icon">✳</span>

              <span>
                <strong>Summarize</strong>
                <small>Concise clinical text</small>
              </span>
            </div>

            <div className="floating-note floating-note-bottom">
              <span className="floating-icon purple-icon">⌕</span>

              <span>
                <strong>Verify</strong>
                <small>Review possible omissions</small>
              </span>
            </div>
          </div>
        </section>

        {/* TRUST STRIP */}
        <div className="trust-strip">
          <span>BUILT FOR CLINICAL TEXT RESEARCH</span>
          <span className="trust-divider" />
          <span>Extractive summarization</span>
          <span className="trust-divider" />
          <span>Rule-based verification</span>
          <span className="trust-divider" />
          <span>Human review required</span>
        </div>

        {/* FEATURES SECTION */}
        <section
          className="features-section section-container"
          id="features"
        >
          <div className="section-intro">
            <div>
              <span className="section-kicker">WHAT IT DOES</span>

              <h2>
                A second look at
                <br />
                clinical information.
              </h2>
            </div>

            <p>
              A research workflow that brings summarization and rule-based
              screening together in one place.
            </p>
          </div>

          <Features />
        </section>

        {/* WORKSPACE SECTION */}
        <section className="workspace-section" id="workspace">
          <div className="section-container">
            <div className="workspace-intro">
              <div>
                <span className="section-kicker">YOUR WORKSPACE</span>

                <h2>From clinical note to review.</h2>

                <p>
                  Enter text, generate a summary, and inspect the screening
                  results returned by the research prototype.
                </p>
              </div>

              <div className="workspace-step">
                <span className="step-number">01</span>
                <span>INPUT → ANALYZE → REVIEW</span>
              </div>
            </div>

            {/* INPUT COMPONENT */}
            <NoteInput
              clinicalNote={clinicalNote}
              setClinicalNote={setClinicalNote}
              numSentences={numSentences}
              setNumSentences={setNumSentences}
              loading={loading}
              error={error}
              onSummarize={handleSummarize}
              onClear={handleClear}
            />

            {/* RESULTS SECTION */}
            {result && (
              <section
                className="results-section"
                id="results"
                aria-live="polite"
              >
                <div className="results-intro">
                  <div>
                    <span className="section-kicker">02 / OUTPUT</span>
                    <h2>Review the results.</h2>
                    <p>
                      Inspect the summary and each rule-based finding.
                    </p>
                  </div>

                  <span className="result-count">
                    {result.verification_count} checks returned
                  </span>
                </div>

                {/* GENERATED SUMMARY */}
                <article className="result-card summary-result-card">
                  <div className="result-card-heading">
                    <span className="result-icon">✳</span>

                    <div>
                      <h3>Generated summary</h3>
                      <p>Review against the original source note.</p>
                    </div>
                  </div>

                  <p className="summary-text">
                    {result.summary || "No summary was generated."}
                  </p>
                </article>

                {/* METRICS */}
                <div className="metrics-grid">
                  <article className="metric-card">
                    <span className="metric-label">
                      Verification checks
                    </span>
                    <strong>{result.verification_count}</strong>
                    <small>Configured terms checked</small>
                  </article>

                  <article className="metric-card">
                    <span className="metric-label">
                      Possible review flags
                    </span>
                    <strong>{result.possible_omission_count}</strong>
                    <small>Require human inspection</small>
                  </article>
                </div>

                {/* VERIFICATION TABLE */}
                <article className="result-card">
                  <div className="result-card-heading">
                    <span className="result-icon purple-result-icon">
                      ⌕
                    </span>

                    <div>
                      <h3>Verification findings</h3>
                      <p>
                        Rule-based checks are not a clinical accuracy score.
                      </p>
                    </div>
                  </div>

                  {!Array.isArray(result.verification) ||
                  result.verification.length === 0 ? (
                    <p className="empty-state">
                      No configured terms were returned for verification.
                      This does not mean the summary is complete.
                    </p>
                  ) : (
                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>Term</th>
                            <th>Category</th>
                            <th>Source status</th>
                            <th>Summary status</th>
                            <th>Finding</th>
                            <th>Priority</th>
                          </tr>
                        </thead>

                        <tbody>
                          {result.verification.map((item, index) => (
                            <tr key={`${item.Term}-${index}`}>
                              <td className="term-cell">{item.Term}</td>
                              <td>{item.Category}</td>
                              <td>{item["Source Status"]}</td>
                              <td>{item["Summary Status"]}</td>
                              <td>
                                <span className="finding-tag">
                                  {item.Verification}
                                </span>
                              </td>
                              <td>{item.Priority}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </article>

                {/* POSSIBLE OMISSIONS */}
                <article className="result-card">
                  <div className="result-card-heading">
                    <span className="result-icon warning-result-icon">
                      !
                    </span>

                    <div>
                      <h3>Possible omissions</h3>
                      <p>
                        Screening flags that require human review.
                      </p>
                    </div>

                    <span className="omission-count">
                      {result.possible_omission_count}
                    </span>
                  </div>

                  {!Array.isArray(result.possible_omissions) ||
                  result.possible_omissions.length === 0 ? (
                    <p className="empty-state">
                      No possible omissions were flagged by the current rules.
                      This does not establish that the summary is complete or
                      safe.
                    </p>
                  ) : (
                    <div className="omission-list">
                      {result.possible_omissions.map((item, index) => (
                        <div
                          className="omission-item"
                          key={`${item.Term}-${index}`}
                        >
                          <div>
                            <strong>{item.Term}</strong>
                            <span>{item.Category}</span>
                          </div>

                          <span className="finding-tag">
                            {item.Verification}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </article>

                {/* RESEARCH DISCLAIMER */}
                <div className="disclaimer">
                  <strong>
                    Research prototype — human review required
                  </strong>

                  <p>
                    {result.disclaimer ||
                      "This tool is experimental and is not clinically validated. It must not be used as the sole basis for diagnosis, treatment, or discharge decisions."}
                  </p>
                </div>
              </section>
            )}
          </div>
        </section>

        {/* ABOUT SECTION */}
        <section className="about-section section-container" id="about">
          <div className="about-mark">+</div>

          <div>
            <span className="section-kicker">ABOUT THE PROJECT</span>
            <h2>Built to support careful review.</h2>

            <p>
              SafeDischargeAI is a research prototype exploring clinical text
              summarization and rule-based screening for potentially omitted
              information. Its outputs need independent human assessment.
            </p>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="page-footer">
        <a className="footer-brand" href="#home">
          <span className="brand-icon">+</span>
          <span>
            SafeDischarge<span className="brand-ai">AI</span>
          </span>
        </a>

        <span>Clinical text summarization &amp; verification</span>

        <a href="#home">Back to top ↑</a>
      </footer>
    </div>
  );
}

export default App;