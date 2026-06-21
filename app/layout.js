import { Baloo_2, Geist_Mono } from "next/font/google";
import "./globals.css";

const baloo = Baloo_2({
  variable: "--font-baloo",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "ResNet50 ONNX — Browser Image Classifier",
  description:
    "Classify images in the browser with ResNet50 exported to ONNX and ONNX Runtime Web.",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${baloo.variable} ${geistMono.variable} h-full antialiased`}
      style={{ "--font-body": "system-ui, sans-serif" }}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
