import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AskProvider } from "@/components/ask/ask-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Ripple — what happens to sales if…?",
  description:
    "Bayesian what-if simulator and launch tracker. The model makes the numbers. The agent explains them. The web gives context.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <AskProvider>
          {children}
          <footer className="mx-auto w-full max-w-[1200px] px-6 py-6 text-xs text-muted-foreground">
            Synthetic demo data · MIT ·{" "}
            <a href="https://github.com/rezataeb/ripple" className="underline underline-offset-2 hover:text-foreground">
              Source on GitHub
            </a>
          </footer>
        </AskProvider>
      </body>
    </html>
  );
}
