import type { QuestionL10n } from "@/types/exam";
import { arLt } from "./a1a3-aviation-regulations";
import { asLt } from "./a1a3-airspace-restrictions";
import { fsLt } from "./a1a3-flight-safety";
import { hpLt } from "./a1a3-human-performance";
import { opLt } from "./a1a3-operational-procedures";
import { ukLt } from "./a1a3-uas-general-knowledge";
import { prLt } from "./a1a3-privacy-data-protection";
import { insLt } from "./a1a3-insurance";
import { secLt } from "./a1a3-security";

/** Lithuanian text for every A1/A3 question, keyed by question id. */
export const a1a3Lt: Record<string, QuestionL10n> = {
  ...arLt, ...asLt, ...fsLt, ...hpLt, ...opLt, ...ukLt, ...prLt, ...insLt, ...secLt,
};

export const a1a3CategoriesLt: Record<string, string> = {
  "Aviation Regulations":      "Aviacijos reguliavimas",
  "Airspace Restrictions":     "Oro erdvės apribojimai",
  "Flight Safety":             "Skrydžių sauga",
  "Human Performance":         "Žmogaus galimybių ribos",
  "Operational Procedures":    "Veiklos procedūros",
  "UAS General Knowledge":     "Bendrosios žinios apie UAS",
  "Privacy & Data Protection": "Privatumo ir duomenų apsauga",
  "Insurance":                 "Draudimas",
  "Security":                  "Saugumas",
};
