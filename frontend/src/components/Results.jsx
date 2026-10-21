function Results({ result }) {
  if (!result) {
    return null;
  }

  const verification = Array.isArray(result.verification)
    ? result.verification
    : [];

  const possibleOmissions = Array.isArray(result.possible_omissions)
    ? result.possible_omissions
    : [];

  const verificationCount =
    result.verification_count ?? verification.length;

  const omissionCount =
    result.possible_omission_count ?? possibleOmissions.length;

  return (
    <section
      className="results-section"
      id="results"
      aria-live="polite"
    >
      {/* RESULTS HEADER */}
      <div className="results-intro">
        <div>
          <span className="section-kicker">02 / OUTPUT</span>
          <h2>Review the results.</h2>
          <p>Inspect the summary and each rule-based finding.</p>
        </div>

        <span className="result-count">
          {verificationCount} checks returned
        </span>
      </div>

      {/* GENERATED SUMMARY */}
      <article className="result-card summary-result-card">
        <div className="result-card-heading">
          <span className="result-icon" aria-hidden="true">
            ✳
          </span>

          <div>
            <h3>Generated summary</h3>
            <p>Review against the original source note.</p>
          </div>
        </div>

        <p className="summary-text">
          {result.summary || "No summary was generated."}
        </p>
      </article>

      {/* SUMMARY METRICS */}
      <div className="metrics-grid">
        <article className="metric-card">
          <span className="metric-label">
            Verification checks
          </span>

          <strong>{verificationCount}</strong>

          <small>Configured terms checked</small>
        </article>

        <article className="metric-card">
          <span className="metric-label">
            Possible review flags
          </span>

          <strong>{omissionCount}</strong>

          <small>Require human inspection</small>
        </article>
      </div>

      {/* VERIFICATION FINDINGS */}
      <article className="result-card">
        <div className="result-card-heading">
          <span
            className="result-icon purple-result-icon"
            aria-hidden="true"
          >
            ⌕
          </span>

          <div>
            <h3>Verification findings</h3>
            <p>
              Rule-based checks are not a clinical accuracy score.
            </p>
          </div>
        </div>

        {verification.length === 0 ? (
          <p className="empty-state">
            No configured terms were returned for verification. This does
            not mean the summary is complete.
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
                {verification.map((item, index) => (
                  <tr
                    key={`${item.Term ?? "term"}-${index}`}
                  >
                    <td className="term-cell">
                      {item.Term ?? "—"}
                    </td>

                    <td>{item.Category ?? "—"}</td>

                    <td>{item["Source Status"] ?? "—"}</td>

                    <td>{item["Summary Status"] ?? "—"}</td>

                    <td>
                      <span className="finding-tag">
                        {item.Verification ?? "—"}
                      </span>
                    </td>

                    <td>{item.Priority ?? "—"}</td>
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
          <span
            className="result-icon warning-result-icon"
            aria-hidden="true"
          >
            !
          </span>

          <div>
            <h3>Possible omissions</h3>
            <p>Screening flags that require human review.</p>
          </div>

          <span className="omission-count">
            {omissionCount}
          </span>
        </div>

        {possibleOmissions.length === 0 ? (
          <p className="empty-state">
            No possible omissions were flagged by the current rules.
            This does not establish that the summary is complete or safe.
          </p>
        ) : (
          <div className="omission-list">
            {possibleOmissions.map((item, index) => (
              <div
                className="omission-item"
                key={`${item.Term ?? "omission"}-${index}`}
              >
                <div>
                  <strong>
                    {item.Term ?? "Unspecified term"}
                  </strong>

                  <span>
                    {item.Category ?? "Uncategorized"}
                  </span>
                </div>

                <span className="finding-tag">
                  {item.Verification ?? "Requires review"}
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
  );
}

export default Results;