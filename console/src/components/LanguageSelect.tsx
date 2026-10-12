import type { LanguageSelection } from "../hooks/useLanguage";
import { SourceTag } from "./SourceTag";

// The language a per-language page is working on (see hooks/useLanguage.ts). The options are
// whatever languages the data contains; nothing is listed that the system has no trace of.
export function LanguageSelect({ selection }: { selection: LanguageSelection }) {
  const { languages, language, setLanguage, source } = selection;
  if (!languages) return null;
  if (languages.length === 0) {
    return <span className="language-select">No language has a kiosk, a model or a review item yet.</span>;
  }
  return (
    <>
      <label className="language-select">
        Language
        <select value={language ?? ""} onChange={(e) => setLanguage(e.target.value)}>
          {languages.map((code) => (
            <option key={code} value={code}>
              {code.toUpperCase()}
            </option>
          ))}
        </select>
      </label>
      <SourceTag source={source} label="language list" />
    </>
  );
}
