export interface Question {
  id: string;
  category?: string;
  q: string;
  options: string[];
  /** index into options */
  answer: number;
  explanation?: string;
  /** ⚡ reasoning / scenario question */
  reasoning?: boolean;
}

export interface ExamConfig {
  id: string;
  shortName: string;
  name: string;
  description: string;
  questionCount: number;
  timeMinutes: number;
  passPercent: number;
  color: string;
}

/** Translated text for one question (answer index is shared with the source). */
export interface QuestionL10n {
  q: string;
  options: string[];
  explanation?: string;
}
