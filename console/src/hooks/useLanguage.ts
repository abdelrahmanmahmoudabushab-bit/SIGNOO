import { useCallback, useEffect, useState } from "react";
import { fetchLanguages } from "../api/endpoints";
import type { Language } from "../api/types";

const STORAGE_KEY = "signoo_console_language";

export interface LanguageSelection {
  /** null while loading; may be empty when no language is known yet. */
  languages: Language[] | null;
  /** The one language the page is working on, or null while loading / when none is known. */
  language: Language | null;
  setLanguage: (language: Language) => void;
  source: "server" | "mock" | null;
}

// Every language is its own engine, so the per-language pages work on exactly one at a time.
// The choice is remembered across pages, so moving from the registry to training stays on the
// same language.
export function useLanguage(): LanguageSelection {
  const [languages, setLanguages] = useState<Language[] | null>(null);
  const [language, setLanguageState] = useState<Language | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchLanguages().then(({ data, source: src }) => {
      if (cancelled) return;
      let remembered: string | null = null;
      try {
        remembered = localStorage.getItem(STORAGE_KEY);
      } catch {
        // localStorage unavailable: start on the first language
      }
      setLanguages(data);
      setSource(src);
      setLanguageState(remembered && data.includes(remembered) ? remembered : (data[0] ?? null));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // best-effort only
    }
  }, []);

  return { languages, language, setLanguage, source };
}
