"use client";

import dynamic from "next/dynamic";
import type { Question } from "@/types/exam";

const FlashcardSession = dynamic(() => import("./FlashcardSession"), { ssr: false });

interface Props {
  questions: Question[];
  title: string;
  storagePrefix?: string;
  onBack?: () => void;
}

export default function FlashcardSessionClient({ questions, title, storagePrefix, onBack }: Props) {
  return <FlashcardSession questions={questions} title={title} storagePrefix={storagePrefix} onBack={onBack} />;
}
