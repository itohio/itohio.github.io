"use client";

import dynamic from "next/dynamic";
import type { Question, ExamConfig } from "@/types/exam";

const ExamSession = dynamic(() => import("./ExamSession"), { ssr: false });

interface Props {
  questions: Question[];
  config: ExamConfig;
  onBack?: () => void;
}

export default function ExamSessionClient({ questions, config, onBack }: Props) {
  return <ExamSession questions={questions} config={config} onBack={onBack} />;
}
