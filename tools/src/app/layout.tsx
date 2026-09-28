import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ITOHI Tools — browser instruments for FPV, telemetry and hardware",
  description: "Interactive analysis tools from ITOHI. Everything runs client-side in the browser — no upload, no account, your logs never leave your machine. EASA A1/A3 and A2 exam prep in English and Lithuanian.",
  robots: "index, follow",
  openGraph: {
    title: "ITOHI Tools — browser instruments for FPV, telemetry and hardware",
    description: "EASA A1/A3 + A2 CoC exam prep (EN/LT), pre-flight checklists, RX blind-spot viewer. Everything runs in your browser.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
