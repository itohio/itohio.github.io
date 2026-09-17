import type { Question } from "@/types/exam";
import { aviationRegulationsQuestions } from "./aviation-regulations";
import { airspaceRestrictionsQuestions } from "./airspace-restrictions";
import { flightSafetyQuestions } from "./flight-safety";
import { humanPerformanceQuestions } from "./human-performance";
import { operationalProceduresQuestions } from "./operational-procedures";
import { uasGeneralKnowledgeQuestions } from "./uas-general-knowledge";
import { privacyDataProtectionQuestions } from "./privacy-data-protection";
import { insuranceQuestions } from "./insurance";
import { securityQuestions } from "./security";

export const a1a3Questions: Question[] = [
  ...aviationRegulationsQuestions,
  ...airspaceRestrictionsQuestions,
  ...flightSafetyQuestions,
  ...humanPerformanceQuestions,
  ...operationalProceduresQuestions,
  ...uasGeneralKnowledgeQuestions,
  ...privacyDataProtectionQuestions,
  ...insuranceQuestions,
  ...securityQuestions,
];

export const a1a3Categories = [
  { key: "Aviation Regulations",       lt: "Aviacijos reguliavimas",        count: aviationRegulationsQuestions.length },
  { key: "Airspace Restrictions",      lt: "Oro erdvės apribojimai",        count: airspaceRestrictionsQuestions.length },
  { key: "Flight Safety",              lt: "Skrydžių sauga",                count: flightSafetyQuestions.length },
  { key: "Human Performance",          lt: "Žmogaus galimybių ribos",       count: humanPerformanceQuestions.length },
  { key: "Operational Procedures",     lt: "Veiklos procedūros",            count: operationalProceduresQuestions.length },
  { key: "UAS General Knowledge",      lt: "Bendrosios žinios apie UAS",    count: uasGeneralKnowledgeQuestions.length },
  { key: "Privacy & Data Protection",  lt: "Privatumo ir duomenų apsauga",  count: privacyDataProtectionQuestions.length },
  { key: "Insurance",                  lt: "Draudimas",                     count: insuranceQuestions.length },
  { key: "Security",                   lt: "Saugumas",                      count: securityQuestions.length },
];
