import Link from "next/link";
import { getSortedPostsData } from "../../lib/api";
import { Scroll, Github } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function BlogPage() {
  const allPostsData = getSortedPostsData();

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <Link href="/blog" className="flex items-center">
            <Scroll className="w-10 h-10 mr-2 text-primary" strokeWidth={1.5} />
            <span className="text-2xl font-bold">docetl blog</span>
          </Link>
          <Button asChild variant="ghost" className="flex items-center">
            <a
              href="https://github.com/ucbepic/docetl"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-gray-600 hover:text-gray-900"
            >
              <Github className="w-6 h-6 mr-2" />
              <span>GitHub</span>
            </a>
          </Button>
        </div>
        <h1 className="text-3xl font-bold mb-8">Latest Posts</h1>
        <ul className="space-y-8">
          {allPostsData.map(({ id, date, title }) => (
            <li key={id} className="border-b pb-6">
              <Link
                href={`/blog/${id}`}
                className="text-2xl font-semibold hover:text-blue-500 block mb-2"
              >
                {title}
              </Link>
              <span className="text-gray-500">{date}</span>
            </li>
          ))}
        </ul>
        <div className="mt-12">
          <Link href="/" className="text-blue-500 hover:underline">
            &larr; Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
