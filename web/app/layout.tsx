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
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <span>Synthetic demo data · MIT</span>
              <a href="mailto:taeb.reza@gmail.com" className="underline underline-offset-2 hover:text-foreground">
                Contact
              </a>
              <a
                href="https://github.com/rezataeb/ripple"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2 hover:text-foreground"
              >
                GitHub
              </a>
            </div>
          </footer>
        </AskProvider>
      </body>
    </html>
  );
}
