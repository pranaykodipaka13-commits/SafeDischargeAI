
function NoteInput({
  clinicalNote,
  setClinicalNote,
  numSentences,
  setNumSentences,
  loading,
  error,
  onSummarize,
  onClear,
}) {
  return (
    <section className="workspace-card">
      <div className="workspace-card-header">
        <div className="workspace-card-title">
          <span className="workspace-symbol">+</span>
          <div>
            <h3>Clinical note analyzer</h3>
            <p>Summarization and omission screening</p>
          </div>
        </div>
        <span className="demo-badge">RESEARCH MODE</span>
      </div>

      <div className="workspace-card-body">
        <div className="section-heading">
          <div>
            <span className="step-label">01 / INPUT</span>
            <h3>Source clinical note</h3>
            <p>Paste synthetic or de-identified clinical text.</p>
          </div>
          <span className="privacy-label">Text input</span>
        </div>

        <label className="visually-hidden" htmlFor="clinical-note">
          Source clinical note
        </label>

        <textarea
          id="clinical-note"
          className="note-input"
          value={clinicalNote}
          onChange={(event) => setClinicalNote(event.target.value)}
          placeholder="Paste a clinical note here to begin your review..."
          maxLength={100000}
          rows={9}
          aria-describedby="note-help"
        />

        <div className="input-footer">
          <span id="note-help">
            {clinicalNote.length.toLocaleString()} / 100,000 characters
          </span>

          <label className="sentence-setting">
            Summary length
            <select
              value={numSentences}
              onChange={(event) => setNumSentences(event.target.value)}
            >
              <option value={3}>3 sentences</option>
              <option value={5}>5 sentences</option>
              <option value={8}>8 sentences</option>
              <option value={10}>10 sentences</option>
              <option value={15}>15 sentences</option>
            </select>
          </label>
        </div>

        <div className="actions">
          <button
            className="primary-button"
            onClick={onSummarize}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="button-spinner" />
                Analyzing note...
              </>
            ) : (
              <>
                Generate summary <span aria-hidden="true">→</span>
              </>
            )}
          </button>

          <button
            className="secondary-button"
            onClick={onClear}
            disabled={loading}
          >
            Clear note
          </button>
        </div>

        {error && (
          <div className="error-message" role="alert">
            <strong>Unable to complete the request</strong>
            <p>{error}</p>
          </div>
        )}
      </div>
    </section>
  );
}

export default NoteInput;
